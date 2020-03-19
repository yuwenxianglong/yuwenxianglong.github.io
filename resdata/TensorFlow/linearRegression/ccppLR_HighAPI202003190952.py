# -*- coding: utf-8 -*-
"""
@Project : linearRegression
@Author  : Xu-Shan Zhao
@Filename: ccppLR_HighAPI202003190952.py
@IDE     : PyCharm
@Time1   : 2020-03-19 09:52:23
@Time2   : 2020/3/19 9:52
@Month1  : 3月
@Month2  : 三月
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

csv_data = pd.read_csv('Folds5x2_pp.csv')
fts = csv_data.iloc[:, 0: 4].astype('float32').to_numpy()
pe = csv_data.iloc[:, 4].astype('float32').to_numpy()

model = tf.keras.layers.Dense(units=1)

# plt.scatter(pe, model(fts))
# plt.show()

num_epoches = 100
learning_rate = 0.1

for epoch in range(num_epoches):
    with tf.GradientTape() as tape:
        y_ = model(fts)
        loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(pe, y_))

    grads = tape.gradient(loss, model.variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(grads, model.variables))

    print(epoch + 1, '/', num_epoches, loss.numpy())

plt.scatter(pe, model(fts))
plt.show()
