---
title: DeePMD-kit的安装和自我启蒙
author: 赵旭山
tags: 随笔



---

笔者认为，大数据技术、机器学习技术是目前材料信息技术能够得以工程应用的最高效模式，工作之余摸索了不少相关开源代码，为避免一两年后把这些学习历程忘得一干二净，随笔点滴记录一下。

> 点滴记录，幼儿学步，不是教程。

DeePMD-kit是基于神经网络模型拟合多体势的一个优秀软件，基于TensorFlow机器学习框架开发。难能可贵的是，作者及时跟进Tensorflow 2.0版本的更新，对代码进行了迁移和更新，跟进最近科学进展的前瞻性很让笔者佩服。

#### 1. 安装篇

几个月前用conda安装过一次，但当时的代码不兼容TensorFlow 2.0，又特别不适应“`conda create -n your_env_name python=X.X`”虚拟环境这种“混乱”方式，所以简单折腾了一下放弃了。此次趁着在家抗疫的空闲，在自己的MacOS 10.15.3和Ubuntu 19.10（虚拟机）上测试进行了安装，成功运行。

笔者习惯于以conda（Miniconda）为主管理python环境，如非必要不使用pip。所以以下安装记录基于conda进行。

conda提供了DeePMD-kit的安装Source，包括以下Packages。但遗憾的是，此Source仅支持`linux-64`系统。

![](/assets/images/condaDeepmodelingSource202002221313.png)

<img src="/assets/images/deepmdOnlySupportLinux202002221318.png" style="zoom:50%;" />

##### **MacOS**

conda的“`conda forge`” channel 提供了`deepmd-kit`、`dpdata`和`dpgen`软件包，但却没有提供lammps-dp，所以deepmd-kit所得的势函数（`xxxx.pb`），是无法被常规lammps程序（conda或brew安装）读取的。

```
deepmd-kit                     1.1.3  py37h2af55cb_1  conda-forge
dpdata                        0.1.15            py_0  conda-forge
dpgen                          0.7.0            py_0  conda-forge
```

故执行：`conda install deepmd-kit dpdata dpgen -c conda-forge`即可完成安装。

##### Ubuntu

ubuntu下deepmd-kit安装比较简单，依照官方“Readme”文件即可。

```bash
conda install deepmd-kit lammps-dp dpdata dpgen -c deepmodeling
```

没有安装GPU版本，有机会以后试试。