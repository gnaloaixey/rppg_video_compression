# Experiment
##
The project only publishes the project framework, and the algorithms for individual research reproduction are not uploaded and published.
## dataset loaer



## 运行

### 安装 conda，创建环境

```
conda create -n compression python=3.8.18
conda activate compression
```
### 安装显卡驱动相应的 cuda、cuDNN

### 根据 cuda 安装相应 pytorch 的版本

<a href='https://pytorch.org/get-started/previous-versions/'>pytorch version</a>

```
# CUDA 10.2
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
```

### 安装其他依赖

```
# dlib
conda install -c conda-forge dlib
# 其他依赖
conda env update --file .\compression.yml
```
### 编译安装支持的 ffmpeg
