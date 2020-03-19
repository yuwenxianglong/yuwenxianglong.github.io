---
title: TensorFlow之线性回归：低阶API实现
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---

> 《[TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)》文中分别通过**低阶API**、**高级API**、**Keras API**三种方式实现了**单变量**（一元）线性回归。
>
> 但笔者参考此文，使用[前文](https://yuwenxianglong.github.io/2020/03/02/Pandas%E8%AF%BB%E5%8F%96CSV%E6%95%B0%E6%8D%AE%E8%BD%AC%E5%8C%96%E4%B8%BAPyTorch%E5%BC%A0%E9%87%8F.html)发电厂运行数据集，进行**多变量**（多元）线性回归时，低阶API和高级API两种方式效果都不好。
>
> 但也记录一下**低阶API**回归的过程，希望随着学习的深入，写出更完善的代码。



本文利用[前文](https://yuwenxianglong.github.io/2020/03/02/Pandas%E8%AF%BB%E5%8F%96CSV%E6%95%B0%E6%8D%AE%E8%BD%AC%E5%8C%96%E4%B8%BAPyTorch%E5%BC%A0%E9%87%8F.html)中使用的发电厂运行数据集，使用TensorFlow 2.1.0版本，应用低阶API实现回归分析。代码主要参考《[TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)》一文。

#### 1. Pandas读取csv数据

```python
data_csv = pd.read_csv('Folds5x2_pp.csv')
fts = data_csv.iloc[:, 0: 4].to_numpy().astype('float32')
pe = data_csv.iloc[:, 4].to_numpy().astype('float32')
```

需注意的是，特征向量`fts`和目标向量`pe`均需通过`astype`实现将dataframe字段类型转换为`float32`型，否则`tf.matmul`操作会报错：

```python
tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute MatMul as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:MatMul] name: MatMul/
```

#### 2. 定义网络

```python
class Model(object):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform([4, 1]))
        self.b = tf.Variable(tf.random.uniform([1]))

    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)
```

用类似[PyTorch八股文](https://yuwenxianglong.github.io/2020/02/22/%E5%85%AB%E8%82%A1%E6%96%87%E5%AE%9A%E4%B9%89Pytorch%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.html)的方式定义网络，但还是不一样，`__init__`函数仅定义了变量，`__call__`函数定义了线性方程。

真正类似[PyTorch八股文](https://yuwenxianglong.github.io/2020/02/22/%E5%85%AB%E8%82%A1%E6%96%87%E5%AE%9A%E4%B9%89Pytorch%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.html)的方式随后高阶API中会提及。

#### 3. 实例化网络模型并定义损失函数

```python
learning_rate = 0.1    

model = Model()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def loss_fn(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(tf.square(y_ - y))
```

本数据集用TensorFlow线性回归与[PyTorch线性回归](https://yuwenxianglong.github.io/2020/03/03/PyTorch%E5%A4%9A%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.html)相同，使用**梯度下降优化器**模型训练不收敛，改用Adam优化器后效果尚可。

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
```

#### 4. 模型训练

```python
num_epoches = 300

for epoch in range(num_epoches):
    with tf.GradientTape() as tape:  # 追踪梯度
        loss = loss_fn(model, fts, pe)  # 计算损失
    grads = tape.gradient(loss, [model.W, model.b])  # 计算梯度
    optimizer.apply_gradients(grads_and_vars=zip(grads, [model.W, model.b]))  # 更新梯度
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)
    print(epoch + 1, '/', num_epoches, loss.numpy(), '\n', model.W.numpy(), '\n', model.b.numpy())
```

`tf.GradientTape`是模型训练的关键，用于梯度追踪。

> “TensorFlow 2.0 中的 Eager Execution 提供了 `tf.GradientTape` 用于追踪梯度。”

Adam优化器更新梯度：

```python
optimizer.apply_gradients(grads_and_vars=zip(grads, [model.W, model.b]))
```

梯度下降优化器更新梯度，但对本文中数据集不适用，所以弃用了。

```python
# model.W.assign_sub(learning_rate * dW)
# model.b.assign_sub(learning_rate * db)
```

#### 5. `tf.GradientTape`

`tf.GradientTape`函数实现自动求导功能。

> “tensorflow 提供tf.GradientTape api来实现自动求导功能。只要在tf.GradientTape()上下文中执行的操作，都会被记录与“tape”中，然后tensorflow使用反向自动微分来计算相关操作的梯度。”

官网文档上给出的示例：

```python
# 求一阶导数
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # 函数y对自变量x求一阶导数，值为6.0
```

```python
# 求二阶导数
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
```

```python
# 隐函数求导
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = g.gradient(y, x)  # 6.0
del g  # Drop the reference to the tape
```















#### References:

* [TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)
* [TensorFlow 2.0 基础：张量、自动求导与优化器](https://blog.csdn.net/zkbaba/article/details/100060157)
* [tf.GradientTape](https://tensorflow.google.cn/api_docs/python/tf/GradientTape?hl=zh-CN)
* [TensorFlow2.0教程-自动求导](https://zhuanlan.zhihu.com/p/69951925)

