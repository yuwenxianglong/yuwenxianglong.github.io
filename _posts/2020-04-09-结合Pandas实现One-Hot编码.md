---
title: 结合Pandas实现”One-Hot Encoding“
author: 赵旭山
tags: 随笔
typora-root-url: ..
---



前文讲解过如果利用`tensorflow.feature_column`进行"One-Hot"编码，Pandas的`get_dummies`函数也提供了相应的功能。

```python
data = pd.read_csv('heart.csv')
data = pd.get_dummies(data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
```

输出为：

```
In [12]: data                                                                                                                                                               
Out[12]: 
     age  trestbps  chol  thalach  oldpeak  sex_0  sex_1  cp_0  cp_1  cp_2  ...  slope_3  ca_0  ca_1  ca_2  ca_3  thal_1  thal_2  thal_fixed  thal_normal  thal_reversible
0     63       145   233      150      2.3      0      1     0     1     0  ...        1     1     0     0     0       0       0           1            0                0
1     67       160   286      108      1.5      0      1     0     0     0  ...        0     0     0     0     1       0       0           0            1                0
2     67       120   229      129      2.6      0      1     0     0     0  ...        0     0     0     1     0       0       0           0            0                1
3     37       130   250      187      3.5      0      1     0     0     0  ...        1     1     0     0     0       0       0           0            1                0
4     41       130   204      172      1.4      1      0     0     0     1  ...        0     1     0     0     0       0       0           0            1                0
..   ...       ...   ...      ...      ...    ...    ...   ...   ...   ...  ...      ...   ...   ...   ...   ...     ...     ...         ...          ...              ...
298   52       118   186      190      0.0      0      1     0     1     0  ...        0     1     0     0     0       0       0           1            0                0
299   43       132   341      136      3.0      1      0     0     0     0  ...        0     1     0     0     0       0       0           0            0                1
300   65       135   254      127      2.8      0      1     0     0     0  ...        0     0     1     0     0       0       0           0            0                1
301   48       130   256      150      0.0      0      1     0     0     0  ...        0     0     0     1     0       0       0           0            0                1
302   63       150   407      154      4.0      1      0     0     0     0  ...        0     0     0     0     1       0       0           0            0                1

[303 rows x 31 columns]

In [13]: data.columns                                                                                                                                                       
Out[13]: 
Index(['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1',
       'cp_0', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'fbs_0', 'fbs_1', 'restecg_0',
       'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_1', 'slope_2',
       'slope_3', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'thal_1', 'thal_2',
       'thal_fixed', 'thal_normal', 'thal_reversible'],
      dtype='object')
```

以下为构建`tensorflow`模型并开展训练的常规步骤，不再赘述。

```python
y = data.pop('target').values
X = data.values

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=128, activation='relu', input_shape=[X.shape[1]]),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ]
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
history = model.fit(
    x = X,
    y = y,
    shuffle=True,
    batch_size=128,
    validation_split=0.2,
    epochs=500,
    callbacks=[lr_reduce, early_stop]
)

hist = pd.DataFrame(history.history, index=history.epoch)
hist['loss'].plot(legend=True)
hist['val_loss'].plot(legend=True)
plt.show()
hist['accuracy'].plot(legend=True)
hist['val_accuracy'].plot(legend=True)
plt.show()
```







