# -*- coding: utf-8 -*-
"""
@Project : linearRegression
@Author  : Xu-Shan Zhao
@Filename: ccppLR_KerasAPI202003191722.py
@IDE     : PyCharm
@Time1   : 2020-03-19 17:22:21
@Time2   : 2020/3/19 17:22
@Month1  : 3月
@Month2  : 三月
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

csv_data = pd.read_csv('Folds5x2_pp.csv')
fts = csv_data.iloc[:, 0: 4].astype('float32').to_numpy()
pe = csv_data.iloc[:, 4].astype('float32').to_numpy()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=4))

num_epoches = 100
steps_per_epoch = 299

model.compile(optimizer='adam', loss='mse')
model.fit(fts, pe, steps_per_epoch=steps_per_epoch, epochs=num_epoches)

plt.figure()
plt.plot(pe, model(fts), 'ro')
plt.plot(pe, pe)
plt.show()