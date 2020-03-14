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

总结一下：

**参数列表**：

* `input_size`：输入向量`x`的特征维度；
* `hidden_size`：隐藏层的特征维度；
* `num_layers`：LSTM隐藏层的层数，默认为1；
* `bias`：False则$b_i$和$b_h$均为0。默认为True；
* `batch_first`：True则输入输出的数据为（batch, seq, feature)；
* `dropout`：除最后一层，每一层的输出都进行dropout（如丢弃一半，0.5），默认为：0；
* `bidirectional`：True则为双向LSTM，默认为False。

**输入输出数据格式**：

运行模型：

```python
output, (hn, cn) = model(input, (h0, c0))
```

输入数据格式：

`input`：`(seq_len, batch, input_size)`

`h0`：`(num_layers*num_directions, batch, hidden_size)`

`c0`：`(num_layers*num_directions, batch, hidden_size)`

输出数据格式：

`output`：`(seq_len, batch, hidden_size*num_directions)`

`h0`：`(num_layers*num_directions, batch, hidden_size)`

`c0`：`(num_layers*num_directions, batch, hidden_size)`

#### 3. 示例

##### 3.1 简单示例

```python
rnn = nn.LSTM(10, 20, 2)  # 输入向量维数input_size为10，隐藏元维度hidden_size为20，2个LSTM层num_layers串联（默认为1层）

input = torch.randn(5, 3, 10)  # 输入(seq_len, batch, input_size)，序列长度为5，batch为3，输入维度为10

h0 = torch.randn(2, 3, 20)  # h0(num_layers * num_directions, batch, hidden_size)

c0 = torch.randn(2, 3, 20)  # 同上

output, (hn, cn) = rnn(input, (h0, c0))  # 运行模型
```



##### 3.2 示例二

示例来源：[https://zhuanlan.zhihu.com/p/41261640](https://zhuanlan.zhihu.com/p/41261640)

有些地方不是特别懂，但注释很详细。

```python
lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
# 记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
print(lstm.all_weights)

inputs = [torch.randn(1, 3) for _ in range(5)]
# 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
# 每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
# 同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
# 对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。
print('Inputs:', inputs)

# 初始化隐藏状态
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
print('Hidden:', hidden)
for i in inputs:
    # 将序列的元素逐个输入到LSTM，这里的View是把输入放到第三维，看起来有点古怪，
    # 回头看看上面的关于LSTM输入的描述，这是固定的格式，以后无论你什么形式的数据，
    # 都必须放到这个维度。就是在原Tensor的基础之上增加一个序列维和MiniBatch维，
    # 这里可能还会有迷惑，前面的1是什么意思啊，就是一次把这个输入处理完，
    # 在输入的过程中不会输出中间结果，这里注意输入的数据的形状一定要和LSTM定义的输入形状一致。
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)
print('out1:', out)
print('hidden2:', hidden)
# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print('out2', out)
print('hidden3', hidden)
```





#### References

* [牛刀小试之用pytorch实现LSTM](https://mp.weixin.qq.com/s/k_z8sNbO3sqqkTV8gvWaIw)
* [什么是 LSTM 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-B-LSTM/)
* [LSTM:Pytorch实现](https://blog.ddlee.cn/posts/7b4533bb/)
* [总结PYTORCH中nn.lstm(自官方文档整理 包括参数、实例)](https://www.pianshen.com/article/9440379844/)
* [Pytorch的LSTM的理解](https://zhuanlan.zhihu.com/p/41261640)

