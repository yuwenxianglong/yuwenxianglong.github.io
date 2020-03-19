# -*- coding: utf-8 -*-
"""
@Project : linearRegression
@Author  : Xu-Shan Zhao
@Filename: ccppLRLikeTorch202003192351.py
@IDE     : PyCharm
@Time1   : 2020-03-19 23:51:28
@Time2   : 2020/3/19 23:51
@Month1  : 3月
@Month2  : 三月
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

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
