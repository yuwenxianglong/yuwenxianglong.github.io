---
title: TensorFlow之PyTorch八股文式编程
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---

从前，折腾了一年多`TensorFlow 1.X`也没能入门，PyTorch是第一个入门的机器学习框架，得益于其逻辑清晰易懂的代码结构。TensorFlow升级到2.0+以后，深度集成Keras，代码结构甚至比PyTorch更为简单了。

TensorFlow 2.0+提供了一种“**自由度更高的模型**”定义方式，与PyTorch非常类似。

本文将[上文](https://yuwenxianglong.github.io/2020/03/19/TensorFlow%E4%B9%8B%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92KerasAPI%E5%AE%9E%E7%8E%B0.html)中的线性回归模型重新按照PyTorch八股文的方式进行编码。

```python
csv_data = pd.read_csv('Folds5x2_pp.csv')
fts = csv_data.iloc[:, : 4].astype('float32').to_numpy()
pe = csv_data.iloc[:, 4].astype('float32').to_numpy()

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(units=1, input_dim=fts.shape[1])

    def call(self, x):
        x =self.dense_1(x)
        return x

model = Net()

model.compile(optimizer='adam', loss='mse')
model.fit(fts, pe, steps_per_epoch=299, epochs=100)

plt.figure()
plt.scatter(pe, model(fts))
plt.plot(pe, pe)
plt.show()
```



























#### References：

* [TensorFlow 2.0 构建神经网络](https://huhuhang.com/post/machine-learning/tensorflow-2-0-03)