---
title: TensorFlow之保存和重载模型
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---





[上一文](https://yuwenxianglong.github.io/2020/04/17/Tensorflow%E4%B9%8B%E8%AE%A4%E8%AF%86%E5%8D%B7%E9%9B%86%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.html)中采用卷积神经网络分类CIFAR-10数据为基础，阐述几种模型保存和重载的方法：

* 训练中保存Checkpoint，从最新保存的Checkpoint中恢复模型；
* 仅保存模型权重，重载时需先定义网络结构模型，再加载权重。Checkpoint和最终模型保存均适用；
* 保存整个模型，包括网络结构和权重等所有信息；
* `save_model`方式保存模型，此种方式重载模型后可以`predict`，但不能`evaluate`，因为此类重载后的模型还需要`compile`

#### 1. 构建网络结构模型

和前一文相同，不做赘述。

```python
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


model = create_model()
```

#### 2. 保存和加载checkpoint

首先定义保存路径和文件命名格式。

```python
checkpoint_path = 'train\cp-{epoch:04d}.ckpt'  # Used to save checkpoint, point to a certain file.
checkpoint_dir = os.path.dirname(checkpoint_path)  # Used for reloading latest checkpoint, point to a directory.
```

##### 2.1 保存模型初始状态

`epoch=0`传入**chekpoint_path**格式化为待保存的文件名。

```python
In [2]: checkpoint_path.format(epoch=0)
Out[2]: 'train\\cp-0000.ckpt'
```

保存checkpoint：

```python
model.save(checkpoint_path.format(epoch=0))
```

##### 2.2 训练过程中保存checkpoint

通过`callbacks.ModelCheckpoint`定义`fit`过程中保存checkpoint。

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_freq=400000,  # save_freq='epoch', or period=10,
    # save_weights_only=True
)

history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=2560,
    callbacks=[checkpoint_callback],
    validation_split=0.2
)
```

`save_freq=400000`设定保存频率，`save_weights_only=True`设定只保存权重（True）或保存整个模型。

##### 2.3 谈谈`period=10`和`save_freq=400000`

TensorFlow 2.1.0版本提示`period`马上要被废弃，需要用`save_freq`去代替，但其实两个设定的用法是不同的。

`period=10`的含义其实挺明了的，就是每10个`epoch`保存一次。（其实挺通俗易懂的，搞不懂为什么要废弃。）

`save_freq`有两类取值，‘epoch’和整数。‘epoch’比较好懂，就是每一个‘epoch’保存一次。整数则有些费解，测试几次，应该是与样本数量有关系，如`save_freq=40000`似乎是每40,000个样本被送入训练后保存一次，本例中等价于`epoch=10`。

##### 2.4 从checkpoint中加载模型

用到了`tf.train.latest_checkpoint`函数，从保存checkpoint的目录中找到最后一次保存的checkpoint。

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = create_model()
# model.load_weights(latest)
model_fromCP = tf.keras.models.load_model(latest)
model_fromCP.evaluate(train_images, train_labels)
```

其他的无需赘述，如果保存`check_point`时选择了`save_weights_only=True`，那么再加载时：

```python
model = create_model()
model.load_weights(latest)
```

即：先构建出来网络结构模型，再把保存的权重加载进来。

#### 3. 保存和重载整个模型

此处以HDF5文件格式保存模型，可以用HDFView查看。

```python
model.save('cifar10CNN.h5')
model_fromHDF5 = tf.keras.models.load_model('cifar10CNN.h5')
model_fromHDF5.evaluate(train_images, train_labels)
```

#### 4. 仅保存权重

仅保存权重，需要先构建模型。

```python
model.save_weights('weight\manual_weight')
model_fromWeight = create_model()
model_fromWeight.load_weights('weight\manual_weight')
model.evaluate(train_images, train_labels)
```

#### 5. save_model方式保存加载模型

save_model方式保存的模型，重载后可以直接使用`model.predict`进行预测，但是不能`evaluate`，需要进行`compile`。如此看，save_model仅保存了模型结构和权重，但没有保存优化器、损失函数、metrics等信息。详见：《[TensorFlow 2 中文文档 - 保存与加载模型 ](https://geektutu.com/post/tf2doc-ml-basic-save-model.html)》。



#### References:

* [TensorFlow 2 中文文档 - 保存与加载模型 ](https://geektutu.com/post/tf2doc-ml-basic-save-model.html)