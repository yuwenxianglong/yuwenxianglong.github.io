---
title: TensorFlow之线性回归：低阶API实现
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---

本文利用[前文](https://yuwenxianglong.github.io/2020/03/02/Pandas%E8%AF%BB%E5%8F%96CSV%E6%95%B0%E6%8D%AE%E8%BD%AC%E5%8C%96%E4%B8%BAPyTorch%E5%BC%A0%E9%87%8F.html)中使用的发电厂运行数据集，使用TensorFlow 2.1.0版本，应用低阶API实现回归分析。代码主要参考《[TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)》一文。

```python
data_csv = pd.read_csv('Folds5x2_pp.csv')
fts = data_csv.iloc[:, 0: 4].to_numpy().astype('float32')
pe = data_csv.iloc[:, 4].to_numpy().astype('float32')


class Model(object):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform([4, 1]))
        self.b = tf.Variable(tf.random.uniform([1]))

    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)


num_epoches = 300
learning_rate = 0.1    

model = Model()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def loss_fn(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(tf.square(y_ - y))


for epoch in range(num_epoches):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, fts, pe)
    grads = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(grads_and_vars=zip(grads, [model.W, model.b]))
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)
    print(epoch + 1, '/', num_epoches, loss.numpy(), '\n', model.W.numpy(), '\n', model.b.numpy())
```















#### References:

* [TensorFlow 2.0 实现线性回归](https://huhuhang.com/post/machine-learning/tensorflow-2-0-02)

* [TensorFlow 2.0 基础：张量、自动求导与优化器](https://blog.csdn.net/zkbaba/article/details/100060157)