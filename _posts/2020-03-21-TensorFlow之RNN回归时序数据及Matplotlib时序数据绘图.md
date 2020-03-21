---
title: TensorFlow之RNN回归时序数据及Matplolib时序数据绘图
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---



最典型的时序数据莫过于人们热切渴望能够预测未来的股票数据，本文通过Tushare获取每日股票交易的最高价，利用RNN实现回归。

#### 1. 获取时序数据预处理

```python
stockHistData = ts.get_hist_data('601600')
high = stockHistData['high']
high = (high - high.mean()) / (high.max() - high.min())  # 数据归一化处理，[-1, 1]
high = high.sort_index(ascending=True)  # 按日期正序排列
```

此处获取的数据是按照日期倒序排列的，故用Pandas的`sort_index`函数对数据重新排为正序。

#### 2. 定义时序宽度

对此的理解详见[前文](https://yuwenxianglong.github.io/2020/03/16/PyTorch%E4%B9%8BRNN%E6%8B%9F%E5%90%88%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE.html)。

```python
input_size = 30

df = pd.DataFrame()
for i in range(input_size):
    df['c%d' % i] = high.tolist()[i: -input_size + i]

df.loc[len(high) - input_size] = high.tolist()[-input_size:]
df.index = high.index[input_size - 1:]
```

#### 3. 定义网络模型

```python
X = df.iloc[:, :].astype('float32').to_numpy()
X = X.reshape(X.shape[0], 1, X.shape[1])  # 若在此处reshpe则self.layer1不需要
y = high.iloc[input_size - 1:].astype('float32').to_numpy()

class GRUModel(tf.keras.Model):
    def __init__(self):
        super(GRUModel, self).__init__()
        # self.layer1 = tf.keras.layers.Reshape((input_size, 1))  # 前面如果已reshape，此处不需要
        # self.layer11 = tf.keras.layers.GRU(60, return_sequences=True)
        self.layer2 = tf.keras.layers.GRU(60)
        self.layer3 = tf.keras.layers.Dense(1)

    def call(self, x):
        # x = self.layer1(x)
        # x = self.layer11(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```

GRU输入为三维张量：`(batch_size, timesteps/seq_len, input_dim)`。所以须对`X`进行`reshape`操作。以下方法二选一。

* **若采用numpy方式**

参照[前文](https://yuwenxianglong.github.io/2020/03/16/PyTorch%E4%B9%8BRNN%E6%8B%9F%E5%90%88%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE.html)PyTorch中RNN的输入张量，则应为：

```python
X = X.reshape(X.shape[0], 1, X.shape[1])
```

**注意**，PyTorch中RNN[默认](https://yuwenxianglong.github.io/2020/03/14/PyTorch%E4%B9%8BLSTM%E5%87%BD%E6%95%B0.html)输入张量顺序为`(seq_len, batch_size, input_dim)`，而keras中RNN中输入`(batch_size, seq_len, input_dim)`。

但其实`X = X.reshape(X.shape[0], X.shape[1], 1)`也工作，不太明白。

* **若采用keras的Reshape函数**

```python
self.layer1 = tf.keras.layers.Reshape((input_size, 1))
```

但其实`(1, input_size)`也能工作。分别对应于使用numpy的两种方式：或者把`input_size=30`放在1前面，或者放后面。

```python
In [2]: model = tf.keras.layers.Reshape((30, 1))

In [3]: model(X).shape
Out[3]: TensorShape([474, 30, 1])

In [4]: model = tf.keras.layers.Reshape((1, 30))

In [5]: model(X).shape
Out[5]: TensorShape([474, 1, 30])
```

#### 4. 训练及可视化

```python
model = GRUModel()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # 若自定义学习率，需单独定义optimizer
# model.compile(optimizer=optimizer, loss='mse')
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, batch_size=100, epochs=200)

plt.figure(figsize=(19, 9))
ax = plt.gca()
ax.plot(df.index, y, 'o', df.index, model(X), 'r-')
ax.xaxis.set_major_locator(plt.MultipleLocator(36))
plt.xlim(df.index[0], df.index[-1])
plt.xticks(rotation=90)
plt.show()
```

横坐标为日期格式，默认绘图时或者太密（直接用`plt.plot`），或者太稀疏（用Pandas的`df.plot`）。所以使用以下Matplotlib函数定义了横坐标刻度和标签旋转。

```python
ax.xaxis.set_major_locator(plt.MultipleLocator(36))
plt.xticks(rotation=90)
```

#### 5. GRU函数

摘录了Keras官方API文档以供参考。

```python
tf.keras.layers.GRU(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    recurrent_constraint=None, bias_constraint=None, dropout=0.0,
    recurrent_dropout=0.0, implementation=2, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, unroll=False,
    time_major=False, reset_after=True, **kwargs
)
```







#### References：

* [tf.keras.layers.GRU](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRU?hl=zh-cn)
* [Tensorflow 2.0 快速入门 —— RNN 预测牛奶产量](https://www.jianshu.com/p/e2ff67c7b7aa)
* [4_RNN_Many_to_One_TF2_0.ipynb](https://github.com/zht007/tensorflow-practice/blob/master/5_Prediction_MilkProdction/4_RNN_Many_to_One_TF2_0.ipynb)

