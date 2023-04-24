import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import os
from PIL import Image
import matplotlib.pyplot as plt
from model.ca_net_cal import CANet_cal
from utils.attention_zoom import batch_augment
from utils.evaluate import calc_map_k,CenterLoss
# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()
# feature_center: size of (#classes, #attention_maps * #channel_features)
feature_center = torch.zeros(200, 32 * 2048).cuda()
class HInterface(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CANet_cal(self.config)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        # print('pseudocode\n',train_batch)
        x, y, pseudocode = train_batch
        # feats = attention_map
        alpha1, alpha2, f44_b, y33, feats, p, px,feature_maxtrix = self.model(x)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += 5e-2 * (feature_maxtrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            zoom_images = batch_augment(x, feats, mode='zoom')
        _, _, _, y_zoom, _, y_pred_aug, y_pred_aux_aug, _= self.model(zoom_images)
        y_pred_aux = torch.cat([px, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y], dim=0)

        # y_att = (y33 + y_zoom)/2
        # loss_y = smooth_CE(y_att, y, 0.9)
        loss_y = cross_entropy_loss(p, y) / 3. + \
                 cross_entropy_loss(y_pred_aux, y_aux) * 2. / 3. + \
                 cross_entropy_loss(y_pred_aug, y)  / 3. + \
                 center_loss(feature_maxtrix, feature_center_batch)
        loss_code = F.mse_loss(f44_b, pseudocode)
        loss = loss_code * (1 / alpha1) ** 2 + loss_y * (1 / alpha2) ** 2 + \
               torch.log(alpha1 + 1) + torch.log(alpha2 + 1)

        loss = loss.mean()

        self.log('train_loss', loss)
        self.log('alpha1', alpha1)
        self.log('alpha2', alpha2)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, flag = val_batch
        _, _, f44_b, _, _, _, _, _ = self.model(x)

        outputs = {'output_code': f44_b,
                   'label': y,
                   'flag': flag}

        return outputs

    def validation_epoch_end(self, outputs):
        # print('outputs\n',outputs)
        print("%s %d validation end, and calculate the metrics for hashing!" % (
        self.config.dataset, self.config.code_length))
        # flag==0 is gallery, and flag==1 is query
        gallery_code = []
        gallery_label = []
        query_code = []
        query_label = []

        for i in range(len(outputs)):
            flag_gallary = outputs[i]['flag'] == 0
            flag_query = outputs[i]['flag'] == 1

            gallery_code.append(outputs[i]['output_code'][flag_gallary])
            gallery_label.append(outputs[i]['label'][flag_gallary])
            query_code.append(outputs[i]['output_code'][flag_query])
            query_label.append(outputs[i]['label'][flag_query])

        gallery_code = torch.cat(gallery_code)
        gallery_label = torch.cat(gallery_label)
        query_code = torch.cat(query_code)
        query_label = torch.cat(query_label)

        print("gallery_code:", gallery_code.size())
        print("query_code:", query_code.size())

        gallery_onehot = F.one_hot(gallery_label).to(torch.float)
        query_onehot = F.one_hot(query_label).to(torch.float)

        # ret_path = "log/%s_%s/%s_%s" % (self.config.model_name, self.config.dataset, self.config.code_length, self.config.dataset)
        # torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.pth'))
        # torch.save(gallery_code.cpu(), os.path.join(ret_path, 'gallery_code.pth'))
        # torch.save(query_onehot.cpu(), os.path.join(ret_path, 'query_targets.pth'))
        # torch.save(gallery_onehot.cpu(), os.path.join(ret_path, 'gallery_targets.pth'))

        map_1 = calc_map_k(torch.sign(query_code), torch.sign(gallery_code), query_onehot, gallery_onehot)
        # map_1 = calc_map_k(torch.sign(code_q), torch.tensor(train_codes).float().cuda(), labels_onehot_q.cuda(),
        #                    labels_onehot_d.cuda())
        self.log("val_mAP", map_1)
        print("mAP:%f" % map_1)
        return


# Utils (such as smooth_CE and so on)
def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.size()

    label_logits = F.one_hot(label, num_cls)
    smooth_label = torch.ones(logits.size()) * (1 - peak) / (num_cls - 1)
    smooth_label[label_logits == 1] = peak

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label.to(logits.device))
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss

