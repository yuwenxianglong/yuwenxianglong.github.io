---
title: PyTorch之LSTM函数
author: 赵旭山
tags: 随笔
typora-root-url: ..
---



#### 1. LSTM简介

莫烦[教程](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-B-LSTM/)通俗易懂地介绍了“什么是LSTM循环神经网络”，文中形象地比喻了一个“忘记控制”，对应了下图中的$ h_t $和$ c_t $两个参数，$ h_t $表示LSTM的输出结果，$ c_t $表示LSTM调整后的“**记忆**”。

![](/assets/images/lstmNetStructure202003141709.jpg)

Fig. source: [https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw](https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw)



![](/assets/images/pytorchLSTMDescription202003141603.jpg)





#### References

* [牛刀小试之用pytorch实现LSTM](https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw)
* [什么是 LSTM 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-B-LSTM/)