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
python main.py --dataset aircraft --gpu 0, --batch_size=32 --code_length=32 --num_workers=4 --lr 0.035
python main.py --dataset vegfru --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.005
python main.py --dataset nabirds --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008
python main.py --dataset food101 --gpu 0, --batch_size=128 --code_length=32 --num_workers=4  --lr 0.008

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