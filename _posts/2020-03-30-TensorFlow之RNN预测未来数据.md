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

直接通过tushare的API获取的股票价格数据如下所示，按照日期倒序排列。采用`sort_index(ascending=True)`改为正序排列。

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

#### 2. 数据预处理

本文中采用`window`函数进行数据**时序宽度**处理，`.window(window_size, shift=1)`，可以参考[前文](https://yuwenxianglong.github.io/2020/03/16/PyTorch%E4%B9%8BRNN%E6%8B%9F%E5%90%88%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE.html)理解。

```
window_size = 30
def batch_dataset(dataset):
    dataset_batched = dataset.batch(window_size, drop_remainder=True)
    return dataset_batched


ds_data = tf.constant(data_stock.values, dtype=tf.float32)
ds_data = tf.data.Dataset.from_tensor_slices(ds_data).window(window_size, shift=1) \
    .flat_map(batch_dataset)
ds_label = tf.constant(data_stock.values[window_size:], dtype=tf.float32)
ds_label = tf.data.Dataset.from_tensor_slices(ds_label)
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(128).repeat()
```

本文代码用到了`tf.constant`、`from_tensor_slice`、`zip`、`.window`、`.batch`、`repeat`。

`.batch`函数除将数据集分为多个批次（batch）送入训练外，本位还有将**ds_train**数据增加一个维度的作用：`(batch_size, seq_lenth, input_size)`。

`.batch`函数将数据集分为多个批次（batch）后，最后一个`batch`的数据可能凑不足`batch_size`，导致报错：

```python
W tensorflow/core/common_runtime/base_collective_executor.cc:217] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_6]]
2020-03-30 16:07:58.792096: W tensorflow/core/common_runtime/base_collective_executor.cc:217] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
```

> 最后一个`batch`满足`batch_size`要求，其实也会报错……

`repeat`作用在于将数据头尾衔接起来，最后一个`batch`不满足`batch_size`，数据集前面的数据会补足。所以，使用`repeat`函数后，须在`model.fit`中增加`steps_per_epoch`，限制每次迭代的步数。

#### 3. 定义网络结构编译模型

```python
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(13)
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
```

注意，连接多个RNN隐藏层时，除最后一个RNN层，前面的需`return_sequences=True`。

> 《[理解LSTM在keras API中参数return_sequences和return_state](https://blog.csdn.net/u011327333/article/details/78501054)》：
>
> * return_sequences：默认 False。在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。 False 返回单个， true 返回全部。
> * return_state：默认 False。是否返回除输出之外的最后一个状态。

#### 4. 模型训练与保存

keras通过callbacks函数可以定义训练中改变学习率和提前中止。

```python
# 100步后，损失率不变，学习率值减半
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                   patience=100)

# 200步后，损失率不变，提前终止训练
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)

history = model.fit(ds_train, epochs=5000, steps_per_epoch=5,
                    callbacks=[lr_callback, stop_callback])

model.save('stockLSTM')
```

#### 5. 可视化训练结果

```python
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
```

其中，通过`tf.expand_dims(**inputs, axis=**int)`在输入数据增加一个特定（`axis=0`）维度。

![](/assets/images/tfRNNLoss202003301643.png)

![](/assets/images/tfRNNfitting202003301644.png)

上图中点为实际股票价格，线为训练的模型预测的价格。

#### 6. 预测未来数据

```python
# Predict future 100 business days.
dfp = data_stock.copy()
for i in range(100):
    pres = model.predict(tf.constant(tf.expand_dims(dfp.values[-30:], axis=0)))
    dfpres = pd.DataFrame(pres, columns=data_stock.columns)
    dfp = dfp.append(dfpres, ignore_index=True)
dfp[column].plot()
plt.show()
```

预测此后100天的数据：第510~609条数据。

第一次训练预测结果：

![](/assets/images/tfRNNPrediction202003301645.png)

第二次训练预测结果：

![](/assets/images/tfRNNPrediction202003301701.png)

两次预测得到了连个不同的结果，第一次预测此后股票的价格总体平稳，第二次预测此后股票价格总体上涨。所以，单纯考虑时序特征预测股票价格纯属鬼扯。





#### References:

* [1-4,时间序列数据建模流程范例](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/1-4%2C%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E6%B5%81%E7%A8%8B%E8%8C%83%E4%BE%8B.md)
* [理解LSTM在keras API中参数return_sequences和return_state](https://blog.csdn.net/u011327333/article/details/78501054)
* [回调函数使用](https://keras.io/zh/callbacks/)

