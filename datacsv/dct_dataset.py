# Time: 2022-12-20-11-34
# Author: Xianxian Zeng
# Name: fg_dataset.py
# Details: Dataset with Pytorch-Lightning for Fine-graiend Hashing

import os
from torch.utils.data import Dataset
import PIL.Image as Image
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        return image

def default_loader(path):
    return Image.open(path).convert('RGB')
class ToTensor(object):
    def __call__(self, ycbcr_image):
        ycbcr_image = torch.from_numpy(ycbcr_image)
        ycbcr_image = ycbcr_image.permute(2, 0, 1)
        return ycbcr_image

class FG_dataset(Dataset):
    def __init__(self, csv_filename, root_dir=None, config=None, data_type='train', transform=None, loader=default_loader):
        self.root_dir = root_dir
        self.data_type = data_type
        self.filename = csv_filename
        self.transform = transform
        self.loader = loader
        self.config = config
        self.totensor = ToTensor()
        self.resize = Resize(224, 224)
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.normalize = Normalize(mean=self.mean, std=self.std)
        imgs = []
        labels = []
        if isinstance(self.filename, (list,tuple)): # (query, gallery) or (train,)
            for i in range(len(self.filename)):
                data_list = pd.read_csv(self.filename[i])
                for index, row in data_list.iterrows():
                    imgs.append((row['img_path'], row['label'], i))
                    labels.append(row['label'])
        self.imgs = imgs

        if self.config is not None:
            self.cal_pseudo_hashing_code(labels)


    def __getitem__(self, index):
        filename, label, flag = self.imgs[index]
        img_path = os.path.join(self.root_dir, filename)
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32)
        image = self.resize(image)
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image = self.normalize(image)
        img = self.totensor(image)
        ycbcr_image = self.totensor(ycbcr_image)
        if self.config is not None:
            return img,ycbcr_image, label, self.pseudo_code[index]
        else:
            return img,ycbcr_image, label, flag

    def __len__(self):
        return len(self.imgs)   

    # generate pseudo-hashing-code like FISH(TIP2022)
    def cal_pseudo_hashing_code(self, labels):
        bits = self.config.code_length
        with torch.no_grad():
            train_labels = torch.tensor(labels) if isinstance(labels, list) else labels
            # training one-hot code
            tohc = F.one_hot(train_labels).to(torch.float)
            train_size = train_labels.size(0)
            sigma = torch.tensor(1, dtype=torch.float)
            delta = torch.tensor(0.0001, dtype=torch.float)
            setting_iter = 15

            V = torch.randn(bits, train_size)
            B = torch.sign(torch.randn(bits, train_size))
            S1, E, S2 = torch.svd(torch.mm(B, V.t()))
            R = torch.mm(S1, S2)
            T = tohc.t()

            for i in range(setting_iter):
                print("Code generating with %d..." % i)
                B = -1 * torch.ones((bits, train_size))
                B[(torch.mm(R,V)) >= 0] = 1

                Ul = torch.mm(sigma * torch.mm(T, V.t()), 
                              torch.linalg.pinv(sigma * torch.mm(V, V.t())))

                V = torch.mm(torch.linalg.pinv(sigma * torch.mm(Ul.t(), Ul) + delta * torch.mm(R.t(), R)),
                              sigma * torch.mm(Ul.t(), T) + delta * torch.mm(R.t(), B))
                
                S1, E, S2 = torch.svd(torch.mm(B, V.t()))

                R = torch.mm(S1, S2)

            self.pseudo_code = torch.sign(B.t())
            print("Code generated!", "Size:", B.size())
 


