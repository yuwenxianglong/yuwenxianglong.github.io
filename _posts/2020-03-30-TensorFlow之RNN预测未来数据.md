---
title: TensorFlow之RNN预测未来数据
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---

忙着写一份规划，上周没写笔记。穿插写规划之余，认真学习了应用TensorFlow的Keras模块回归和预测时序数据的方法，有所得。

本文的代码主要参考：《[1-4,时间序列数据建模流程范例](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/1-4%2C%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E6%B5%81%E7%A8%8B%E8%8C%83%E4%BE%8B.md)》

#### 1. 获取数据并归一化处理

```python
data_stock = ts.get_hist_data('601600')  # get stock price from tushare
data_stock = data_stock.sort_index(ascending=True)  # sort data by date
data_stock = (data_stock - data_stock.mean()) / (data_stock.max() - data_stock.min())
```



```bash
             open  high  ...      v_ma10      v_ma20
date                    ...
2020-03-30  2.89  2.91  ...   455411.33   586103.73
2020-03-27  2.95  2.96  ...   469928.19   606784.91
2020-03-26  2.96  2.96  ...   488501.13   629158.76
2020-03-25  2.98  2.99  ...   513050.24   647409.67
2020-03-24  2.94  2.95  ...   526814.99   687151.44
...          ...   ...  ...         ...         ...
2018-03-02  5.28  5.43  ...  3320473.30  3320473.30
2018-03-01  5.45  5.64  ...  3090698.50  3090698.50
2018-02-28  5.90  5.90  ...   291075.33   291075.33
2018-02-27  6.55  6.55  ...   133981.00   133981.00
2018-02-26  7.28  7.28  ...   146094.00   146094.00

[509 rows x 13 columns]
```









```python
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

# Compare fitting and real values.
dff = pd.DataFrame()
for i in range(len(data_stock) - window_size):
    fits = model.predict(tf.constant(tf.expand_dims(data_stock.values[i:i + window_size, :], axis=0)))
    dffits = pd.DataFrame(fits, columns=data_stock.columns)
    dff = dff.append(dffits)

dff.index = data_stock.index[window_size:]

plt.figure(figsize=(19, 9))
dff[column].plot()
data_stock.iloc[window_size:, :][column].plot(style='o')
plt.show()

# Predict future 100 business days.
dfp = data_stock.copy()
for i in range(100):
    pres = model.predict(tf.constant(tf.expand_dims(dfp.values[-30:], axis=0)))
    dfpres = pd.DataFrame(pres, columns=data_stock.columns)
    dfp = dfp.append(dfpres, ignore_index=True)
dfp[column].plot()
plt.show()
```











#### References:

[1-4,时间序列数据建模流程范例](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/1-4%2C%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E6%B5%81%E7%A8%8B%E8%8C%83%E4%BE%8B.md)

