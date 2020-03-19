# -*- coding: utf-8 -*-
"""
@Project : linearRegression
@Author  : Xu-Shan Zhao
@Filename: ccppLR_LowAPI202003181956.py
@IDE     : PyCharm
@Time1   : 2020-03-18 19:56:52
@Time2   : 2020/3/18 19:56
@Month1  : 3月
@Month2  : 三月
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

data_csv = pd.read_csv('Folds5x2_pp.csv')
fts = data_csv.iloc[:, 0: 4].to_numpy().astype('float32')
pe = data_csv.iloc[:, 4].to_numpy().astype('float32')

# fts = (fts - fts.mean()) / (fts.max() - fts.min())
# pe = (pe - pe.mean()) / pe.std()


class Model(object):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform([4, 1]))
        self.b = tf.Variable(tf.random.uniform([1]))

    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)


model = Model()

num_epoches = 1000
learning_rate = 10

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def loss_fn(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(tf.square(y_ - y))

plt.figure()
plt.ion

for epoch in range(num_epoches):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, fts, pe)
    grads = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(grads_and_vars=zip(grads, [model.W, model.b]))
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, '/', num_epoches, loss.numpy(), '\n', model.W.numpy(), '\n', model.b.numpy())
        plt.cla()
        plt.scatter(pe, model(fts))
        plt.plot(pe, pe)
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()
