---
title: PyTorch之LSTM函数
author: 赵旭山
tags: 随笔
typora-root-url: ..
---

LSTM是一种循环神经网络，适用于对序列化的输入建模。本文旨在学习PyTorch的LSTM函数，理解输入、输入各参数的意义。

#### 1. LSTM简介

莫烦[教程](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-B-LSTM/)通俗易懂地介绍了“什么是LSTM循环神经网络”，文中形象地比喻了一个“忘记控制”，对应了下图中的$ h_t $和$ C_t $两个参数，$ h_t $表示LSTM的输出结果，$ c_t $表示LSTM调整后的“**记忆**”。

![](/assets/images/lstmNetStructure202003141709.jpg)

Fig. source: [https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw](https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw)

$ X_t $表示输入，**记忆**在网络中的作用如下：

$$ f_t = \sigma (W_f \cdot [h_{t-1}, x_t]  + b_f) $$

$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$

$$ \widetilde{C}_t = tanh(W_c \cdot [h_{t-1}, x_t] + b_c) $$

$$ C_t = f_t \cdot C_{t-1} + i_t \cdot \widetilde{C}_t $$

$$ o_t = \sigma (W_o [h_{t-1}, x_t] + b_o) $$

$$ h_t = o_t \cdot tanh(C_t) $$



#### 2. 输入输出参数及含义

结合PyTorch官方文档中LSTM函数的说明理解各参数。

![](/assets/images/pytorchLSTMDescription202003141603.jpg)

#### 3. 示例

```python
rnn = nn.LSTM(10, 20, 2)  # 输入向量维数input_size为10，隐藏元维度hidden_size为20，2个LSTM层num_layers串联（默认为1层）

input = torch.randn(5, 3, 10)  # 输入(seq_len, batch, input_size)，序列长度为5，batch为3，输入维度为10

h0 = torch.randn(2, 3, 20)  # h0(num_layers * num_directions, batch, hidden_size)

c0 = torch.randn(2, 3, 20)  # 同上

output, (hn, cn) = rnn(input, (h0, c0))
```





#### References

* [牛刀小试之用pytorch实现LSTM](https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw)
* [什么是 LSTM 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-B-LSTM/)
* [LSTM:Pytorch实现](https://blog.ddlee.cn/posts/7b4533bb/)
* [总结PYTORCH中nn.lstm(自官方文档整理 包括参数、实例)](https://www.pianshen.com/article/9440379844/)

