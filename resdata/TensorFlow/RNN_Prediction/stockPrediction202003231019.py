# -*- coding: utf-8 -*-
"""
@Project : RNN_Prediction
@Author  : Xu-Shan Zhao
@Filename: stockPrediction202003231019.py
@IDE     : PyCharm
@Time1   : 2020-03-23 10:19:34
@Time2   : 2020/3/23 10:19
@Month1  : 3月
@Month2  : 三月
"""

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data_stock_ts = ts.get_hist_data('601600')
data_stock_ts = data_stock_ts.sort_index(ascending=True)
data_stock = (data_stock_ts - data_stock_ts.mean()) / (data_stock_ts.max() - data_stock_ts.min())

window_size = 30


def batch_dataset(dataset):
    dataset_batched = dataset.batch(window_size, drop_remainder=True)
    return dataset_batched


ds_train = tf.data.Dataset.from_tensor_slices(tf.constant(data_stock.values, dtype='float32')) \
    .window(window_size, shift=1).flat_map(batch_dataset)

ds_label = tf.data.Dataset.from_tensor_slices(tf.constant(data_stock.values[window_size:], dtype='float32'))

ds_data = tf.data.Dataset.zip((ds_train, ds_label)).batch(128)

model = tf.keras.Sequential(
    [tf.keras.layers.LSTM(128),
     tf.keras.layers.Dense(13)]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(ds_data, epochs=500)

# Using the trained model to predict the future data.
df = data_stock.copy()

# pres_y1 = pres_y0 * (data_stock_ts.max() - data_stock_ts.min()).values + data_stock_ts.mean().values
for i in range(30):
    pres_x0 = df.values[-300:, :]
    pres_x1 = tf.expand_dims(pres_x0, axis=0)
    pres_x2 = tf.constant(pres_x1)
    pres_y0 = model.predict(pres_x2)
    df_pres = pd.DataFrame(pres_y0, columns=df.columns)
    df = df.append(df_pres)

df['ma20'].plot()
plt.show()

model.save('tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('tf_model_savedmodel', compile=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model_loaded.compile(optimizer=optimizer, loss='mse')
# model_loaded.predict(ds_train)
