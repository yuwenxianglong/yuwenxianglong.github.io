---
title: TensorFlow之线性回归：Keras API实现及model.fit()详解
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

```python
num_epoches = 100
steps_per_epoch = 299

model.compile(optimizer='adam', loss='mse')
model.fit(fts, pe, steps_per_epoch=steps_per_epoch, epochs=num_epoches)
```



![](/assets/images/ccppLRKerasAPI202003191755.gif)

#### 4. `model.fit()`

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
```



> 《[Keras model.fit() 函数](https://www.jianshu.com/p/9ba27074044f)》：
>
> - x：输入数据。如果模型只有一个输入，那么x的类型是numpy
>    array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
> - y：标签，numpy array
> - batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
> - epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
> - verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
> - callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
> - validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
> - validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
> - shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
> - class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
> - sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode=’temporal’。
> - initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
>    fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况



> [《steps_per_epoch 与 epochs 的关系](https://blog.csdn.net/hellocsz/article/details/88992039)》：
>
> “拥有越高性能的GPU，则可以设置越大的batch_size值。根据现有硬件，我们设置了每批次输入50-100张图像。参数steps_per_epoch是通过把训练图像的数量除以批次大小得出的。例如，有100张图像且批次大小为50，则steps_per_epoch值为2。参数epoch决定网络中所有图像的训练次数。”









#### References:

* [TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)
* [tf.layers.dense()的用法](https://blog.csdn.net/yangfengling1023/article/details/81774580)
* [TensorFlow函数：tf.layers.Dense](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-rn6a2tps.html)
* [Keras model.fit() 函数](https://www.jianshu.com/p/9ba27074044f)
* [steps_per_epoch 与 epochs 的关系](https://blog.csdn.net/hellocsz/article/details/88992039)

