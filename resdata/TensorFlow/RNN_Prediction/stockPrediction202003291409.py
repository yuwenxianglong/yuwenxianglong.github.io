# -*- coding: utf-8 -*-
"""
@Project : RNN_Prediction
@Author  : Xu-Shan Zhao
@Filename: stockPrediction202003291409.py
@IDE     : PyCharm
@Time1   : 2020-03-29 14:09:49
@Time2   : 2020/3/29 14:09
@Month1  : 3月
@Month2  : 三月
"""

import tensorflow as tf
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

data_stock = ts.get_hist_data('601600')
data_stock = data_stock.sort_index(ascending=True)
data_stock = (data_stock - data_stock.mean()) / (data_stock.max() - data_stock.min())

window_size = 30
column = 'high'


def batch_dataset(dataset):
    dataset_batched = dataset.batch(window_size, drop_remainder=True)
    return dataset_batched


ds_data = tf.constant(data_stock.values, dtype=tf.float32)
ds_data = tf.data.Dataset.from_tensor_slices(ds_data).window(window_size, shift=1) \
    .flat_map(batch_dataset)
ds_label = tf.constant(data_stock.values[window_size:], dtype=tf.float32)
ds_label = tf.data.Dataset.from_tensor_slices(ds_label)
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(128).repeat()

model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(13)
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                   patience=100)
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
history = model.fit(ds_train, epochs=5000, steps_per_epoch=5,
                    callbacks=[lr_callback, stop_callback])

model.save('stockLSTM')

# Plot loss function
plt.figure(figsize=(19, 9))
ax = plt.gca()
plt.plot(range(len(history.history['loss'])), history.history['loss'])
ax.set_yscale('log')
plt.show()

# Compare fitting and real values.
dff = pd.DataFrame()
for i in range(len(data_stock) - window_size):
    fits = model.predict(tf.constant(tf.expand_dims(data_stock.values[i:i + window_size, :], axis=0)))
    dffits = pd.DataFrame(fits, columns=data_stock.columns)
    dff = dff.append(dffits)

dff.index = data_stock.index[window_size:]

plt.figure(figsize=(19, 9))
dff[column].plot()
data_stock.iloc[window_size:, :][column].plot(style='-o')
plt.show()

# To predict future 100 business days.
dfp = data_stock.copy()
for i in range(100):
    pres = model.predict(tf.constant(tf.expand_dims(dfp.values[-30:], axis=0)))
    dfpres = pd.DataFrame(pres, columns=data_stock.columns)
    dfp = dfp.append(dfpres, ignore_index=True)
dfp[column].plot()
plt.show()
