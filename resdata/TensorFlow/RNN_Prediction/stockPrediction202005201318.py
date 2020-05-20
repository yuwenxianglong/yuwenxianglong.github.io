# -*- coding: utf-8 -*-
"""
@Project : RNN_Prediction
@Author  : Xu-Shan Zhao
@Filename: stockPrediction202005201318.py
@IDE     : PyCharm
@Time1   : 2020-05-20 13:18:46
@Time2   : 2020/5/20 13:18
@Month1  : 5月
@Month2  : 五月
"""

import tushare as ts
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

stock_catl = ts.get_hist_data('300750')
stock_catl = stock_catl.sort_index(ascending=True)
stock_catl = (stock_catl - stock_catl.mean()) / \
             (stock_catl.max() - stock_catl.min())

train, val = train_test_split(stock_catl, test_size=0.5)
train = train.sort_index(ascending=True)
val = val.sort_index(ascending=True)

window_size = 7
column = 'high'
epoches = 10


def batch_dataset(dataset):
    dataset_batched = dataset.batch(window_size, drop_remainder=True)
    return dataset_batched


def zip_ds(dataset):
    ds_data = tf.constant(dataset.values, dtype=tf.float32)
    ds_data = tf.data.Dataset.from_tensor_slices(ds_data). \
        window(window_size, shift=1).flat_map(batch_dataset)
    ds_label = tf.constant(dataset.values[window_size:], dtype=tf.float32)
    ds_label = tf.data.Dataset.from_tensor_slices(ds_label)
    ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(128).repeat()
    return ds_train


ds_train = zip_ds(train)
ds_val = zip_ds(val)

model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(13)
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(
    ds_train, epochs=epoches,
    steps_per_epoch=5,
    validation_data=ds_val,
    validation_steps=1
)

model.save('stockLSTM')

# Plot loss function
plt.figure(figsize=(19, 9))
ax = plt.gca()
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'])
ax.set_yscale('log')
plt.show()

# Compare fitting and real values.
dff = pd.DataFrame()
for i in range(len(stock_catl) - window_size):
    fits = model.predict(tf.constant(tf.expand_dims(stock_catl.values[i:i + window_size, :], axis=0)))
    dffits = pd.DataFrame(fits, columns=stock_catl.columns)
    dff = dff.append(dffits)

dff.index = stock_catl.index[window_size:]

plt.figure(figsize=(19, 9))
dff[column].plot()
stock_catl.iloc[window_size:, :][column].plot(style='-o')
plt.show()

# To predict future 100 business days.
dfp = stock_catl.copy()
for i in range(100):
    pres = model.predict(tf.constant(tf.expand_dims(dfp.values[-1 * window_size:], axis=0)))
    dfpres = pd.DataFrame(pres, columns=stock_catl.columns)
    dfp = dfp.append(dfpres, ignore_index=True)
dfp[column].plot()
plt.show()
