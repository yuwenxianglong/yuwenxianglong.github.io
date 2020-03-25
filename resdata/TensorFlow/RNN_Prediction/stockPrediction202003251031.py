# -*- coding: utf-8 -*-
"""
@Project : RNN_Prediction
@Author  : Xu-Shan Zhao
@Filename: stockPrediction202003251031.py
@IDE     : PyCharm
@Time1   : 2020-03-25 10:32:08
@Time2   : 2020/3/25 10:32
@Month1  : 3月
@Month2  : 三月
"""

import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

data_stock = ts.get_hist_data('601600')
data_stock = data_stock.sort_index(ascending=True)
data_stock = (data_stock - data_stock.mean()) / (data_stock.max() - data_stock.min())

window_size = 30
input_size = data_stock.shape[1]


def batch_dataset(dataset):
    dataset_batched = dataset.batch(window_size, drop_remainder=True)
    return dataset_batched


ds_data = tf.constant(data_stock.values, dtype=tf.float32)
ds_data = tf.data.Dataset.from_tensor_slices(ds_data)
ds_data = ds_data.window(window_size, shift=1).flat_map(batch_dataset)
ds_label = tf.data.Dataset.from_tensor_slices(tf.constant(data_stock.values[window_size:], dtype=tf.float32))
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(128).cache()

model = tf.keras.models.Sequential(
    [tf.keras.layers.LSTM(60, return_sequences=True),
     tf.keras.layers.LSTM(60),
     tf.keras.layers.Dense(input_size)]
)

model.compile(optimizer='adam', loss='mse')

logdir = ".\keras_model" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True)
# 如果loss在100个epoch后没有提升，学习率减半。
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=100)
# 当loss在200个epoch后没有提升，则提前终止训练。
stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)
callbacks_list = [tb_callback, lr_callback, stop_callback]

history = model.fit(ds_train, epochs=100, callbacks=callbacks_list)
plt.plot(history.epoch, history.history['loss'], 'bo-')
plt.show()
