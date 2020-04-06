---
title: TensorFlow之特征处理和feature_column
author: 赵旭山
tags: TensorFlow
typora-root-url: ..

---



一直以来，除数据归一化等偏移操作外，对于特征工程朴素的理解就是特征组合。本次有机会通过`feature_column`深入理解了特征工程的其他方面。

本文主要参考《[TensorFlow 2 中文文档 - 特征工程结构化数据分类 ](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)》。

#### 1. 数据来源

数据集来自克利夫兰诊所心脏病基金会（Cleveland Clinic Foundation）提供的[303行14列心脏病数据](https://storage.googleapis.com/applied-dl/heart.csv)，每行描述一个患者，每列代表一个属性，详细的列描述参见[`heart.names`](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)。

以下为本文用到的数据集中各列的说明。

| Column   | Description                                                  | Feature Type   | Data Type |
| -------- | ------------------------------------------------------------ | -------------- | --------- |
| Age      | Age in years                                                 | Numerical      | integer   |
| Sex      | (1 = male; 0 = female)                                       | Categorical    | integer   |
| CP       | Chest pain type (0, 1, 2, 3, 4)                              | Categorical    | integer   |
| Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical      | integer   |
| Chol     | Serum cholesterol in mg/dl                                   | Numerical      | integer   |
| FBS      | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)      | Categorical    | integer   |
| RestECG  | Resting electrocardiographic results (0, 1, 2)               | Categorical    | integer   |
| Thalach  | Maximum heart rate achieved                                  | Numerical      | integer   |
| Exang    | Exercise induced angina (1 = yes; 0 = no)                    | Categorical    | integer   |
| Oldpeak  | ST depression induced by exercise relative to rest           | Numerical      | integer   |
| Slope    | The slope of the peak exercise ST segment                    | Numerical      | float     |
| CA       | Number of major vessels (0~3) colored by flourosopy          | Numerical      | integer   |
| Thal     | 3 = normal; 6 = fixed defect; 7 = reversible defect          | Categorical    | string    |
| Target   | Diagnosis of heart disease (1 = true; 0 = false)             | Classification | integer   |

#### 2. 读入数据分割数据集

使用scimitar-learn的`train_test_split`把数据集分割为：**训练集**、**验证集**和**测试集**。

```python
df = pd.read_csv('heart.csv')

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
```

#### 3. 数据输入管道

之前文中也曾用到`tf.data`创建训练数据集，原来还有这么个“高大上”的包装词，形象的把训练数据包装为“数据帧”。

##### 3.1 创建输入通道

> 《[结构化数据分类实战：心脏病预测(tensorflow2.0官方教程翻译)](https://www.jianshu.com/p/2f08f77593e2)》：
>
> 我们将使用tf.data包装数据帧，这将使我们能够使用特征列作为桥梁从Pandas数据框中的列映射到用于训练模型的特征。如果我们使用非常大的CSV文件（如此之大以至于它不适合内存），我们将使用tf.data直接从磁盘读取它。

> 《[TensorFlow 2 中文文档 - 特征工程结构化数据分类 ](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)》：
>
> 使用 tf.data ，我们可以使用特征工程(feature columns)将 Pandas DataFrame  中的列映射为特征值(features)。如果是一个非常大的 CSV 文件，不能直接放在内存中，就必须直接使用 tf.data  从磁盘中直接读取数据了。

```python
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = df.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(dataframe), labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
tf.print(list(train_ds))
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

与[前文](https://yuwenxianglong.github.io/2020/03/30/TensorFlow%E4%B9%8BRNN%E9%A2%84%E6%B5%8B%E6%9C%AA%E6%9D%A5%E6%95%B0%E6%8D%AE.html)中`tf.data.Dataset.from_tensor_batch(tf.constant(df.values, dtype=tf.float32))`方式创建训练数据集的方式不同，本文中采用：

```python
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(dataframe), labels)
    )
```

的方式，使用了[`dict`函数](https://www.runoob.com/python/python-dictionary.html)。

##### 3.2 理解输入通道

通过`dict`字典方式创建了一个“数据帧”，可以用以下方式查看输入通道返回的数据格式。

```python
for feature_batch, label_batch in train_ds.take(1):
    tf.print(list(feature_batch.keys()))
    tf.print(list(feature_batch['age']))
    tf.print(list(label_batch))
```

#### 4. `tf.feature_column`中几种特征列处理方法

`tf.feature_column`提供了多种特征列处理方法，本节将采用该函数创建几种常用的特征列，以及如何从dataframe中转换。首先定义一个**example_batch**用于后续特征处理。

```python
# fetch one example data
example_ds = next(iter(train_ds))  # using iter and next function, see also reference
example_batch = example_ds[0]  # feature columns
example_label = example_ds[1]  # target column
```

`DenseFeatures`将原始数据转换为特征数据。定义`demo`函数转换特征列并输出数值结果。

```python
def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())
    # tf.print(feature_layer(example_batch))
```

##### 4.1 Numerical columns：数字列

就是数值类型，输出为特征真实的数值。

```python
age = tf.feature_column.numeric_column('age')
demo(age)
```

#### 4.2 Bucketized columns：桶列

> 有时候，并不想直接将数值传给模型，而是希望基于数值的范围离散成几个种类。比如人的年龄，0-10归为一类，用0表示；11-20归为一类，用1表示。我们可以用 bucketized column 将年龄划分到不同的 bucket  中。用中文比喻，就好像提供了不同的桶，在某一范围内的扔进A桶，另一范围的数据扔进B桶，以此类推。下面的例子使用独热编码来表示不同的  bucket。

```python
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)
```

输出为：

```python
[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
```

`boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]`通过**10**个边界值把年龄划分为**11**个区间，故输出了“one-hot”编码的11列数据。

##### 4.3 Categorical columns：分类列

> 在这个数据集中，`thal`列使用字符串表示(e.g. ‘fixed’, ‘normal’,  ‘reversible’)。字符串不能直接传给模型。所以我们要先将字符串映射为数值。可以使用categorical_column_with_vocabulary_list 和 categorical_column_with_vocabulary_file 来转换，前者接受一个列表作为输入，后者可以传入一个文件。

```python
thal = tf.feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = tf.feature_column.indicator_column(thal)
demo(thal_one_hot)
```

输出为：

```python
[[0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
```

把`'fixed', 'normal', 'reversible'`采用“one-hot”编码映射为三列数据。

##### 4.4 Embedding columns：嵌入列

> 假设某一列有上千种类别，用独热编码来表示就不太合适了。这时候，可以使用 embedding column。embedding column 可以压缩维度，因此向量中的值不再只由0或1组成，可以包含任何数字。
>
> 在有很多种类别时使用 embedding column 是最合适的。接下来只是一个示例，不管输入有多少种可能性，最终的输出向量定长为8。

```python
# embedding_column的输入是categorical column
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)
```

输出为：

```python
[[ 0.14364715 -0.2109683  -0.33291674 -0.22627169  0.63493127  0.10321641 -0.4196678  -0.05259528]
 [ 0.14364715 -0.2109683  -0.33291674 -0.22627169  0.63493127  0.10321641 -0.4196678  -0.05259528]
 [ 0.14364715 -0.2109683  -0.33291674 -0.22627169  0.63493127  0.10321641 -0.4196678  -0.05259528]
 [ 0.14364715 -0.2109683  -0.33291674 -0.22627169  0.63493127  0.10321641 -0.4196678  -0.05259528]
 [ 0.14364715 -0.2109683  -0.33291674 -0.22627169  0.63493127  0.10321641 -0.4196678  -0.05259528]]
```

`dimension=8`，故输出为8列数据。

##### 4.5 Hashed feature columns：哈希特征列

> 另一种表示类别很多的 categorical column 的方式是使用 categorical_column_with_hash_bucket。这个特征列会计算输入的哈希值，然后根据哈希值对字符串进行编码。哈希桶(bucket)个数即参数`hash_bucket_size`。哈希桶(hash_buckets)的个数应明显小于实际的类别个数，以节省空间。
>
> 注意：哈希的一大副作用是可能存在冲突，不同的字符串可能映射到相同的哈希桶中。不过，在某些数据集，这个方式还是非常有效的。

```python
thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=20
)
demo(tf.feature_column.indicator_column(thal_hashed))
```

输出为：

```python
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

`hash_bucket_size=20`，故输出为20列。

##### 4.6 Crossed feature columns：交叉特征列

> 将几个特征组合成一个特征，即 feature crosses，模型可以对每一个特征组合学习独立的权重。接下来，我们将组合 `age` 和 `thal` 列创建一个新的特征。注意：`crossed_column`不会创建所有可能的组合，因为组合可能性会非常多。背后是通过`hashed_column`处理的，可以设置哈希桶的大小。

```python
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=16)
demo(tf.feature_column.indicator_column(crossed_feature))
```

输出如下，共计16列。

```python
[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
```





```python
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

# Select columns to be used
feature_columns = []
# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# Define Net sturcture
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'],
              run_eagerly=True)
model.fit(train_ds,
          validation_data=val_ds,
          epochs=5
          )
loss, accuracy = model.evaluate(test_ds)
print(loss, accuracy)
```





#### References:

* [TensorFlow 2 中文文档 - 特征工程结构化数据分类 ](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)
* [结构化数据分类实战：心脏病预测(tensorflow2.0官方教程翻译)](https://www.jianshu.com/p/2f08f77593e2)
* [Python  字典(Dictionary)](https://www.runoob.com/python/python-dictionary.html)
* [tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en)
* [对python中的iter()函数与next()函数详解](https://www.jb51.net/article/149090.htm)

