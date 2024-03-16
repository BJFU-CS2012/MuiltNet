#! /bin/bash
echo '---------------------------创建环境'
mkdir /hy-tmp/workspace
cp /hy-public/CUB2002011/CUB_200_2011.tgz  /hy-tmp/workspace
echo '---------------------------移动文件结束'
cd /hy-tmp/workspace
echo pwd
echo '---------------------------解压文件'
tar -zxvf CUB_200_2011.tgz
echo '---------------------------安装环境'
apt-get install -y libglib2.0-dev
pip install pyTorch-Lightning==1.9.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch_dct -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
echo '---------------------------安装完成'
cd /hy-tmp/Muilti-master/