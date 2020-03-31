---
title: TensorFlow之结合Pandas特征和数据处理
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---



Pandas给笔者的感觉就是一个在内存中操作的SQL数据库，可以为机器学习特征向量预处理和结果后处理提供不少帮助，前面的文中也大量调用了Pandas。

本文代码主要参考：《[TensorFlow 2 中文文档 - 回归预测燃油效率](https://geektutu.com/post/tf2doc-ml-basic-regression.html)》



#### 1. 获取并读入数据

数据集来源于“[Auto MPG Data Set](https://archive.ics.uci.edu/ml/datasets/auto+mpg)”，共计9列398行，数据格式预览如下图。其中，“$ \cdot $”代表一个空格，“$ \longrightarrow $”代表一个`Tab`制表符。

![](/assets/images/autoMPGPreview202003312227.jpg)

此9列数据分别为MPG（city-cycle fuel consumption in **M**iles **P**er **G**allon）、cylinders（气缸数量）、displacement（排量）、马力（horsepower）、重量（weight）、加速度（acceleration）、车型年份（model year）、产地（origin）、车名（car name）。MPG为目标值。

```python
# Download dataset to local folder
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = tf.keras.utils.get_file('auto-mpg.data', url)

# read data using Pandas
column_names = ['MPG', '气缸', '排量', '马力', '重量', '加速度', '年份', '产地']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)  # drop "car name" using comment='t'
dataset = raw_dataset.copy()
# view top 3 data
print(dataset.head(3))
```

#### 2. 数据预处理

##### 2.1 丢弃缺失值

源数据中有6行“horsepower”值缺失，通过`dropna()`函数丢弃包含缺失值的所有行。

```python
print(len(dataset))  # total 398 rows.
print(dataset.isna().sum())  # 6 nan values of horsepower
# drop NA values
dataset = dataset.dropna()
print(len(dataset))  # then 392 rows left
```

##### 2.2 One-Hot Encoding

数据集中“origin(产地)”是三维的：`['美国', '欧洲', '日本']`，为了更便于机器学习算法“理解”这些算法，可以采用“One-Hot”方式进行编码，那么：“美国”就对应`[1, 0, 0]`，“欧洲”就对应`[0, 1, 0]`，“日本”就对应`[0, 0, 1]`。

```python
origin = dataset.pop('产地')  # popup '产地' column and delete it from dataset
dataset['美国'] = (origin == 1) * 1.0  # One-Hot encode 'origin' and concat to dataset
dataset['欧洲'] = (origin == 2) * 1.0
dataset['日本'] = (origin == 3) * 1.0
print(dataset)
```

`pop()`函数“弹出”相应的列，并将此列从原数据集中删除。“One-Hot”编码后，数据集变得如下图所示，最后三列。

```python
      MPG  气缸     排量     马力      重量   加速度  年份   美国   欧洲   日本
0    18.0   8  307.0  130.0  3504.0  12.0  70  1.0  0.0  0.0
1    15.0   8  350.0  165.0  3693.0  11.5  70  1.0  0.0  0.0
2    18.0   8  318.0  150.0  3436.0  11.0  70  1.0  0.0  0.0
3    16.0   8  304.0  150.0  3433.0  12.0  70  1.0  0.0  0.0
4    17.0   8  302.0  140.0  3449.0  10.5  70  1.0  0.0  0.0
..    ...  ..    ...    ...     ...   ...  ..  ...  ...  ...
393  27.0   4  140.0   86.0  2790.0  15.6  82  1.0  0.0  0.0
394  44.0   4   97.0   52.0  2130.0  24.6  82  0.0  1.0  0.0
395  32.0   4  135.0   84.0  2295.0  11.6  82  1.0  0.0  0.0
396  28.0   4  120.0   79.0  2625.0  18.6  82  1.0  0.0  0.0
397  31.0   4  119.0   82.0  2720.0  19.4  82  1.0  0.0  0.0
```

##### 2.3 拆分数据集为训练集和测试集

使用Pandas的`sample`函数“**随机选取若干行**”分割训练集和测试集，详细用法见：《[pandas.DataFrame.sample 随机选取若干行](https://blog.csdn.net/zhengxu25689/article/details/87347700)》和《[Pandas.DataFrame.sample学习](https://zhuanlan.zhihu.com/p/38255793)》。

```python
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```

`frac`表示要选取样本的数量，`random_state`设定随机种子。

##### 2.4 分离特征向量与目标值

同样使用`pop`函数。

```python
# split label
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
```

##### 2.5 可视化向量之间关联

左对角线表示该向量的分布，其他图沿对角线对称，可视化了各特征向量之间的数值关联。

```python
import seaborn as sns

# solve Chinese character gash problem. 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sns.pairplot(train_dataset[['MPG', '气缸', '排量', '重量']], diag_kind='kde')
plt.show()
```

![](/assets/images/pairplot202003312331.jpeg)

##### 2.6 数据归一化

把数据归一化到`[-1, 1]`之间。

```python
def norm(input_ds):
    return (input_ds - input_ds.mean()) / (input_ds.max() - input_ds.min())


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
```

#### 3. 模型构建与编译

`input_shape`选项应该是不必要的。`metrics`会返回训练过程中的评价函数，如果不设置则只会返回`loss`。

```python
input_dim = len(normed_train_data.keys())
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation='relu', input_shape=[input_dim, ]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='mse', metrics=['mae', 'mse'])
print(model.summary())
```

#### 4. 模型训练与可视化

##### 4.1 模型训练——训练集

`callbacks`自动降低学习率、提前终止训练，[前文](https://yuwenxianglong.github.io/2020/03/30/TensorFlow%E4%B9%8BRNN%E9%A2%84%E6%B5%8B%E6%9C%AA%E6%9D%A5%E6%95%B0%E6%8D%AE.html)已有说明，不再赘述。

```python
EPOCHS = 1000
batch_size = len(normed_train_data)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, batch_size=batch_size,
                    callbacks=[lr_reduce, early_stop], verbose=1)
```

可以把训练过程中的评价函数写入Pandas的`DataFrame`中，便于格式化存取和可视化。

```python
# Store training process into Pandas DataFrame
hist = pd.DataFrame(history.history)
print(hist)
```

格式化后的`hist`表如下：

```python
           loss        mae         mse    val_loss    val_mae     val_mse     lr
0    598.030334  23.216547  598.030334  618.799011  23.626965  618.799011  0.001
1    593.163879  23.109701  593.163879  614.951233  23.544111  614.951233  0.001
2    589.512207  23.029507  589.512207  611.483032  23.470024  611.483032  0.001
3    586.237183  22.958015  586.237183  608.198669  23.399837  608.198669  0.001
4    583.176147  22.890944  583.176147  605.079956  23.332827  605.079956  0.001
..          ...        ...         ...         ...        ...         ...    ...
995    5.045089   1.508783    5.045089    7.170265   1.996722    7.170265  0.001
996    5.040422   1.541826    5.040422    7.259096   2.071816    7.259096  0.001
997    5.033629   1.507981    5.033629    7.166165   2.002546    7.166165  0.001
998    5.022903   1.536428    5.022903    7.230778   2.063807    7.230778  0.001
999    5.019257   1.507817    5.019257    7.189934   2.010454    7.189934  0.001
```

使用Pandas内置函数可视化`loss`：

```python
hist['loss'].plot(legend=True, style='-')
hist['val_loss'].plot(legend=True, style=':')
plt.show()
```

![](/assets/images/trainingloss202003312357.jpeg)

##### 4.2 模型验证——验证集

使用`evaluate`函数：

```python
# assess model using test data
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=1)  # 78 test data
print(loss, mae, mse)
```

返回的是损失函数，或其他预定义的`metrics`：

```python
[1000 rows x 7 columns]
78/78 [==============================] - 0s 256us/sample - loss: 5.5059 - mae: 1.8586 - mse: 5.5059
5.505931903154422 1.8585624 5.505932
```

##### 4.3 模型预测——测试集

使用`predict`函数，返回的是预测结果。

```python
# predict MPG data using test data
test_pred = model.predict(normed_test_data).flatten()
print(test_pred)

plt.plot(test_labels, test_pred, 'o')
plt.plot(test_labels, test_labels)
# ax = plt.gca()
# ax.axis('equal')
plt.show()
```

![](/assets/images/autoMPGPrediction202003312358.jpeg)

#### References：

* [TensorFlow 2 中文文档 - 回归预测燃油效率](https://geektutu.com/post/tf2doc-ml-basic-regression.html)
* [Auto MPG Data Set](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
* [数据处理——One-Hot Encoding](https://blog.csdn.net/google19890102/article/details/44039761)
* [pandas.DataFrame.sample 随机选取若干行](https://blog.csdn.net/zhengxu25689/article/details/87347700)
* [Pandas.DataFrame.sample学习](https://zhuanlan.zhihu.com/p/38255793)
* [评价函数的用法](https://keras.io/zh/metrics/)
* [训练集，验证集，测试集](https://blog.csdn.net/qq_40597317/article/details/80639289)
* [训练集、验证集和测试集](https://zhuanlan.zhihu.com/p/48976706)
* [训练集、验证集和测试集的意义](https://www.jianshu.com/p/7e032a8aaad5)