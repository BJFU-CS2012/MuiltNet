Fine-grained Image Retrieval
--------------------------
This is the Official Pytorch-Lightning implementation of the paper: "Cascading Hierarchical Networks with Multi-task
Balanced Loss for Fine-grained hashing"(https://arxiv.org/abs/2303.11274). 
## Requirements
* Python3
* PyTorch
* pip install pyTorch-Lightning == 1.9.4
* Numpy
* pandas


--------------------------
## Dataset
We use the following 5 datasets: CUB200-2011, Aircraft, VegFru, NABirds and Food101.

--------------------------
## RUN

- The running commands for several datasets are shown below. 
```
python main.py --dataset cub --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
python main.py --dataset aircraft --gpu 0, --batch_size=32 --code_length=32 --num_workers=4 --lr 0.0035
python main.py --dataset vegfru --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.005
python main.py --dataset nabirds --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
python main.py --dataset food101 --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
"https://download.pytorch.org/models/resnet50-0676ba61.pth"
python main.py --dataset cub --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
```


--------------------------
## Citation
If you find our work inspiring or use our codebase in your research, please cite our work.
```
@misc{zeng2023cascading,
      title={Cascading Hierarchical Networks with Multi-task Balanced Loss for Fine-grained hashing}, 
      author={Xianxian Zeng and Yanjun Zheng},
      year={2023},
      eprint={2303.11274},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 代码运行指令
```
mkdir /hy-tmp/workspace
cp /hy-public/CUB2002011/CUB_200_2011.tgz  /hy-tmp/workspace
cd /hy-tmp/workspace
tar -zxvf CUB_200_2011.tgz

cd /hy-tmp/FG-CNET-master/
pip install pyTorch-Lightning==1.9.4
pip install opencv-python

chmod 777 build.sh

```

## 实验结果
取后64位：0.8074
取全部位：0.8007
修改为GELU：0.8093
修改了output：0.8093

原始：0.819

## 实验训练
2023-0612 森林人 master中测试使用原始resnet50结构叠加eca模块测试tree数据集

aircraft使用自己的模型：
12bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
24bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
32bit:
森林人：batch-size = 32,lr = 0.0035: 0.8579
恒源云：batch-size = 32,lr = 0.0035: 0.8409
48bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
aircraft使用原始模型：
12bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
24bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
32bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035: 
48bit:
森林人：batch-size = 32,lr = 0.0035: 
恒源云：batch-size = 32,lr = 0.0035:

cub使用自己的模型：
12bit:
森林人：batch-size = 128,lr = 0.008: 
恒源云：batch-size = 128,lr = 0.008: 0.7805
24bit:
森林人：batch-size = 128,lr = 0.008:
恒源云：
32bit:
森林人：batch-size = 128,lr = 0.008:
恒源云：batch-size = 128,lr = 0.008: 0.8412
cub使用原始模型：
12bit:
森林人：batch-size = 128,lr = 0.008: 
恒源云：batch-size = 128,lr = 0.008: 0.7805
24bit:
森林人：batch-size = 128,lr = 0.008:
恒源云：
32bit:
森林人：batch-size = 128,lr = 0.008:
恒源云：batch-size = 128,lr = 0.008: 0.8412


---------------目前实验--------------------------
恒源云：
双输入模型，将裁剪的图像直接用作加强图像 0.8201
双输入模型，将裁剪的图像融合高斯变换后用作加强图像 0.8177
双输入模型，裁剪图像归为16 并融合高斯变换：0.7402
森林人mine：
原始模型融合eca机制,feats = f3_ori ,0.8346
dct 模型融合eca机制,feats = f3_ori
森林人origin:
使用原始模型训练aircraft 检查实验效果:83.7
-------------实验设计----------------------------
1、将图像修改为336大小 8240
2、取消raw分支 8283
3、修改损失函数