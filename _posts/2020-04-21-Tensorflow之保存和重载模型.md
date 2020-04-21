---
title: TensorFlow之保存和重载模型
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---





[上一文](https://yuwenxianglong.github.io/2020/04/17/Tensorflow%E4%B9%8B%E8%AE%A4%E8%AF%86%E5%8D%B7%E9%9B%86%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.html)中采用卷积神经网络分类CIFAR-10数据为基础，阐述几种模型保存和重载的方法：

* 训练中保存Checkpoint，从最新保存的Checkpoint中恢复模型；
* 仅保存模型权重，重载时需先定义网络结构模型，再加载权重。Checkpoint和最终模型保存均适用；
* 保存整个模型，包括网络结构和权重等所有信息；
* `save_model`方式保存模型，此种方式重载模型后可以`predict`，但不能`evaluate`，因为此类重载后的模型还需要`compile`



