---
title: TensorFlow之线性回归：Keras API实现
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---



#### 1. 读入数据

```python
csv_data = pd.read_csv('Folds5x2_pp.csv')
fts = csv_data.iloc[:, 0: 4].astype('float32').to_numpy()
pe = csv_data.iloc[:, 4].astype('float32').to_numpy()
```

#### 2. 构建网路模型

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=4))
```

采用Sequential顺序函数构建网络模型。之后添加一个线性层`tf.keras.layers.Dense()`。

```python
tf.layers.dense(

    inputs,  # 输入该网络层的数据

    units,  # 输出的维度大小，改变inputs的最后一维

    activation=None,  # 激活函数，即神经网络的非线性变化

    use_bias=True,  # 使用bias为True（默认使用），不用bias改成False即可，是否使用偏置项

    kernel_initializer=None,  # 卷积核的初始化器

    bias_initializer=tf.zeros_initializer(),  # 偏置项的初始化器，默认初始化为0

    kernel_regularizer=None,  # 卷积核的正则化，可选

    bias_regularizer=None,  # 偏置项的正则化，可选

    activity_regularizer=None,  # 输出的正则化函数

    kernel_constraint=None,  # 由Optimizer更新后应用于内核的可选投影函数(例如,用于实现层权重的范数约束或值约束).该函数必须将未投影的变量作为输入,并且必须返回投影变量(必须具有相同的形状).在进行异步分布式训练时,使用约束是不安全的。

    bias_constraint=None,  # 由Optimizer更新后应用于偏置的可选投影函数。

    trainable=True,  # 表明该层的参数是否参与训练。如果为True则变量加入到图集合中

    name=None,  # 层的名字

    reuse=None  # 是否重复使用参数

)
```

#### 3. 编译模型开始训练

参数一目了然，不展开解释了。

其中`step_per_epoch`表示每一次迭代送入的样本量(batch_size)。

```python
num_epoches = 100
steps_per_epoch = 299

model.compile(optimizer='adam', loss='mse')
model.fit(fts, pe, steps_per_epoch=steps_per_epoch, epochs=num_epoches)
```

![](/assets/images/ccppLRKerasAPI202003191755.gif)







#### References:

* [TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)

* [tf.layers.dense()的用法](https://blog.csdn.net/yangfengling1023/article/details/81774580)
* [TensorFlow函数：tf.layers.Dense](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-rn6a2tps.html)