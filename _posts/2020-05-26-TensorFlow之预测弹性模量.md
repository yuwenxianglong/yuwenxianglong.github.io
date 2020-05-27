---
title: TensorFlow之预测Bulk modulus
author: 赵旭山
tags: 随笔 matminer TensorFlow
typora-root-url: ..
---



本文利用TensorFlow构建神经网络模型，预测化合物的弹性模量。特征向量的提取与[前文](https://yuwenxianglong.github.io/2020/05/25/Sklearn%E4%B9%8B%E9%A2%84%E6%B5%8B%E5%BC%B9%E6%80%A7%E6%A8%A1%E9%87%8F.html)相同，而后通过`df.to_csv()`保存csv格式文件供本文调用。

> Windows 10下ElementProperty函数无法正常工作，可能与matminer 0.6.3版本Bug有关，所以本例中把上文的特征向量保存为csv文件再在Windows下加载，以使用GPU版本的TensorFlow开展模型训练。

#### 1. 读取并处理特征向量

```python
df = pd.read_csv('elastic_tensor.csv', index_col=0)
# df = df.sort_values(by='K_VRH', ascending=True)
y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula",
            "poisson_ratio", "structure", "composition", "composition_oxid",
            'G_Reuss', 'G_Voigt', 'K_Reuss', 'K_Voigt', 'compliance_tensor',
            'elastic_tensor', 'elastic_tensor_original']
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

#### 2. 构建模型并训练

```python

try:
    model = tf.keras.models.load_model('elasticPres.h5')
except OSError:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=60, input_dim=len(X_train.columns), activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=60, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ]
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mse',
              metrics=['mae', 'mse'])

history = model.fit(X_train, y_train, epochs=1000,
                    batch_size=128, validation_split=0.2)
model.save('elasticPres.h5')

hist = pd.DataFrame(history.history, index=history.epoch)
```

TensorFlow 2.0.0会报如下Warning：

```python
WARNING:tensorflow:Falling back from v2 loop because of error: Failed to find data adapter that can handle input: <class 'pandas.core.frame.DataFrame'>, <class 'NoneType'>
```

TensorFlow 2.1.0不会报。

最后10步的训练结果如下：

```python
Epoch 4991/5000
755/755 [==============================] - 0s 32us/sample - loss: 0.9133 - mae: 0.6431 - mse: 0.9133 - val_loss: 291.3222 - val_mae: 11.8175 - val_mse: 291.3222
Epoch 4992/5000
755/755 [==============================] - 0s 30us/sample - loss: 0.8966 - mae: 0.6300 - mse: 0.8966 - val_loss: 289.3552 - val_mae: 11.7826 - val_mse: 289.3552
Epoch 4993/5000
755/755 [==============================] - 0s 29us/sample - loss: 0.7864 - mae: 0.5899 - mse: 0.7864 - val_loss: 292.4638 - val_mae: 11.9599 - val_mse: 292.4638
Epoch 4994/5000
755/755 [==============================] - 0s 29us/sample - loss: 0.7121 - mae: 0.5586 - mse: 0.7121 - val_loss: 289.2971 - val_mae: 11.8120 - val_mse: 289.2971
Epoch 4995/5000
755/755 [==============================] - 0s 29us/sample - loss: 0.7206 - mae: 0.5556 - mse: 0.7206 - val_loss: 291.6976 - val_mae: 11.8284 - val_mse: 291.6976
Epoch 4996/5000
755/755 [==============================] - 0s 29us/sample - loss: 0.9680 - mae: 0.6612 - mse: 0.9680 - val_loss: 289.1350 - val_mae: 11.8645 - val_mse: 289.1350
Epoch 4997/5000
755/755 [==============================] - 0s 29us/sample - loss: 1.2871 - mae: 0.7952 - mse: 1.2871 - val_loss: 293.5512 - val_mae: 11.9366 - val_mse: 293.5512
Epoch 4998/5000
755/755 [==============================] - 0s 29us/sample - loss: 1.1446 - mae: 0.7474 - mse: 1.1446 - val_loss: 288.7805 - val_mae: 11.7558 - val_mse: 288.7805
Epoch 4999/5000
755/755 [==============================] - 0s 29us/sample - loss: 1.0113 - mae: 0.6959 - mse: 1.0113 - val_loss: 292.6437 - val_mae: 11.9833 - val_mse: 292.6437
Epoch 5000/5000
755/755 [==============================] - 0s 30us/sample - loss: 0.8537 - mae: 0.6302 - mse: 0.8537 - val_loss: 290.2824 - val_mae: 11.8138 - val_mse: 290.2824
```





损失函数收敛情况如下图：

 <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/resdata/MatInf/matminer/elasticPres/Loss.html" height="525" width="100%"></iframe>

模型在训练数据集上的表现，横坐标为DFT计算的体弹性模量值，纵坐标为预测值。

 <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/resdata/MatInf/matminer/elasticPres/BulkPredictionOfTrainingData.html" height="800" width="100%"></iframe>

模型在测试数据集上的表现：

 <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/resdata/MatInf/matminer/elasticPres/BulkPredictionOfTestData.html" height="800" width="100%"></iframe>

