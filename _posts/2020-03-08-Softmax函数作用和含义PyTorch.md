---
title: Softmax函数作用和含义(PyTorch)
author: 赵旭山
tags: PyTorch
---

#### 1. Softmax函数的作用

两个数`a`和`b`，且`a > b`，如果取`max`，那么就直接取用`a`，没有第二种可能。但有时这种取法并不合适，会造成分值小的值永远不可能被取到。

某些时候希望除分值大的项经常被取到外，分值小的项也能偶尔被取到。那么，就需要用到Softmax函数，通过Softmax来计算取`a`和`b`的概率。

#### 2. 定义

Softmax用于多分类过程中，它将多个神经元的输出，映射到`(0, 1)`区间内且和为1，可以看成概率来理解。

对于`n`维向量，Softmax函数定义如下：

$$ f_i(x) = \frac{e^{(x_i-shift)}}{\sum_{j=1}^n e^{(x_j-shift)}},shift=max(x_i) $$

形象的表达如下图：

![](/assets/images/softmax202003081717.jpg)

> "**Softmax直白来说就是将原来输出是3,1,-3通过softmax函数一作用，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标！**"

#### 3. 用法

```python
torch.nn.functional.softmax(input, dim)
```

`dim`：指明维度，`dim=0`表示按列计算；`dim=1`表示按行计算。

```python
In [4]: input                                                   
Out[4]: 
tensor([[ 1.1920, -2.1101,  2.1733],
        [-0.3203,  0.4651,  0.2894],
        [ 0.7082,  0.1558,  1.0614],
        [-0.5396, -0.1877, -0.9587]])

In [5]: softmax(input, dim=0) # 每一列加和为1
Out[5]: 
tensor([[0.4966, 0.0327, 0.6559],
        [0.1094, 0.4291, 0.0997],
        [0.3061, 0.3149, 0.2158],
        [0.0879, 0.2234, 0.0286]])

In [6]: softmax(input, dim=1) # 每一行加和为1
Out[6]: 
tensor([[0.2699, 0.0099, 0.7201],
        [0.1987, 0.4358, 0.3656],
        [0.3334, 0.1919, 0.4747],
        [0.3247, 0.4617, 0.2136]])
```





#### References

[Softmax 函数的特点和作用是什么？](https://www.zhihu.com/question/23765351)

[详解softmax函数以及相关求导过程](https://zhuanlan.zhihu.com/p/25723112)

[torch.nn.functional中softmax的作用及其参数说明](https://www.cnblogs.com/wanghui-garcia/p/10675588.html)

