---
title: Pandas读取CSV数据转化为PyTorch张量
author: 赵旭山
tags: Pytorch
---

#### 1. 获取数据

参考：[https://www.cnblogs.com/pinard/p/6016029.html](https://www.cnblogs.com/pinard/p/6016029.html)

UCI大学公开的机器学习数据集：[http://archive.ics.uci.edu/ml/datasets.php](http://archive.ics.uci.edu/ml/datasets.php)。

本文使用的数据介绍：[http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

下载地址：[http://archive.ics.uci.edu/ml/machine-learning-databases/00294/](http://archive.ics.uci.edu/ml/machine-learning-databases/00294/)

该数据为发电厂的运行状况数据，共计9568个样本，分为5列，分别为：

* AT：温度；
* V：压力
* AP：湿度
* RH：压强
* PE：输出电力

AT、V、AP、RH为特征值，PE为目标值。根据“[用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html)”一文描述，目标值和特征值为线性对应关系，即：

<img src="http://latex.codecogs.com/gif.latex?\ PE = \theta_0 + \theta_1*AT + \theta_2*V + \theta_3*AP + \theta_4*RH" />

<img src="http://latex.codecogs.com/gif.latex?\theta_0" />、<img src="http://latex.codecogs.com/gif.latex?\theta_1" />、<img src="http://latex.codecogs.com/gif.latex?\theta_2" />、<img src="http://latex.codecogs.com/gif.latex?\theta_3" />、<img src="http://latex.codecogs.com/gif.latex?\theta_4" />为需要学习的参数。





#### References

* [用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html)

