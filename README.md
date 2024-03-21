Fine-grained Image Retrieval
--------------------------

*“Multi-FusNet: Fusion mapping of features for fine-grained image retrieval networks”*

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
python main.py --dataset cub --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
```


--------------------------
## 代码运行指令
```
mkdir /hy-tmp/workspace
cp /hy-public/CUB2002011/CUB_200_2011.tgz  /hy-tmp/workspace
cd /hy-tmp/workspace
tar -zxvf CUB_200_2011.tgz

cd /hy-tmp/Muilti-master/
pip install pyTorch-Lightning==1.9.4
pip install opencv-python

chmod 777 build.sh

```

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

