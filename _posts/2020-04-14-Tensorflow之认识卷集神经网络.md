---
title: TensorFlow之认识卷集神经网络
author: 赵旭山
tags: 随笔
typora-root-url: ..
---





工作中很少涉及图像处理，所以一直对故意去忽略卷集神经网络。但是，无论看什么深度学习的书或资料，都架不住卷集神经网络经常在眼前晃悠。学习段代码，姑且算认识吧。



本文代码主要参考《[TensorFlow 2 中文文档 - 卷积神经网络分类 CIFAR-10](https://geektutu.com/post/tf2doc-cnn-cifar10.html)》



#### 1. 数据来源

CIFAR-10数据集，学材料的其实对这个无感，但最近几次参加研究生答辩都经常听到这个CIFAR数据集，对于验证机器学习算法模型应该是比较经典的。既然“**10**”，其实是这些图片样本R/G/B三个通道，每个像素点的颜色由R/G/B三个值决定，R/G/B的取值范围为0～255。据说，熟悉计算机视觉的专业计算机技术人员都知道，其实图片像素点的值由R/G/B/A，A代表透明度，取值范围为0～1。反正我是不知道，学习了。



#### 2. 获取CIFAR-10数据集





#### References：

* [TensorFlow 2 中文文档 - 卷积神经网络分类 CIFAR-10](https://geektutu.com/post/tf2doc-cnn-cifar10.html)

