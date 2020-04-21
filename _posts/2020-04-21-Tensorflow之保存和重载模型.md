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

#### 2.1 保存模型初始状态

`epoch=0`传入**chekpoint_path**格式化为待保存的文件名。

```python
In [2]: checkpoint_path.format(epoch=0)
Out[2]: 'train\\cp-0000.ckpt'
```

保存checkpoint：

```python
model.save(checkpoint_path.format(epoch=0))
```

通过`callbacks.ModelCheckpoint`定义`fit`过程中保存checkpoint。

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_freq=10,
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

`save_freq=10`设定保存频率，`save_weights_only=True`设定只保存权重（True）或保存整个模型。