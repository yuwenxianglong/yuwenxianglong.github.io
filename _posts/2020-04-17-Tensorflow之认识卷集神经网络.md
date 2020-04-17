---
title: TensorFlow之认识卷集神经网络
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---



工作中很少涉及图像处理，所以一直对故意去忽略卷集神经网络。但是，无论看什么深度学习的书或资料，都架不住卷集神经网络经常在眼前晃悠。学习段代码，姑且算认识吧。



本文代码主要参考《[TensorFlow 2 中文文档 - 卷积神经网络分类 CIFAR-10](https://geektutu.com/post/tf2doc-cnn-cifar10.html)》



#### 1. 数据来源

CIFAR-10数据集，学材料的其实对这个无感，但最近几次参加研究生答辩都经常听到这个CIFAR数据集，对于验证机器学习算法模型应该是比较经典的。CIFAR-10数据集共包括60,000张图片数据，分为10类，如下图所示：

![](/assets/images/cifar10CNNDescr202004171836.png)

这些图片样本有R/G/B三个通道（color_channels），即每个像素点的颜色由R/G/B三个值决定，R/G/B的取值范围为0～255。据说“熟悉计算机视觉的专业计算机技术人员都知道，其实图片像素点的值由R/G/B/A，A代表透明度，取值范围为0～1”。反正我是不知道，学习了。

#### 2. 获取CIFAR-10数据集

图片大小为$ 32 \times 32 $像素，标签是整数0~9，与上图10个类别分别对应。

通过keras的`datasets`加载数据集并分割为训练集和测试集。

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
print(train_images.shape)  # (50000, 32, 32, 3)
print(train_labels.shape)  # (50000, 1)
print(test_images.shape)  # (10000, 32, 32, 3)
print(test_labels.shape)  # (10000, 1)
```

#### 2. 数据预处理

图片大小为$ 32 \times 32 $像素，每个像素点取值$ [0, 255] $。转换到$ [0, 1] $范围内后，训练速度明显加快，原因目前笔者还解释不了。训练集和测试集均需转换：

```python
train_images = train_images / 255.
test_images = test_images / 255.
```

#### 3. 卷积网络模型构建

卷积神经网络模型包括：卷积层、池化层和全连接层三类。

```python
model = tf.keras.Sequential(
    [
        # 1st section: convolutional layer & pooling layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 2nd section: fully connected layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)
```

##### 3.1 卷积层和池化层

> CNN 的输入是三维张量 (image_height, image_width, color_channels)，即 input_shape。每一层卷积层使用`tf.keras.layers.Conv2D`来搭建。Conv2D 共接收2个参数，第2个参数是卷积核大小，第1个参数是卷积核的个数。
>
> 卷积层后紧跟了最大池化层(MaxPooling2D)，最大池化即选择图像区域的最大值作为该区域池化后的值，另一个常见的池化操作是平均池化，即计算图像区域的平均值作为该区域池化后的值。

![](/assets/images/cnnStructureDescr202004171914.png)

> 每一轮卷积或池化后，图像的宽和高的值都会减小，假设图像的高度为h，卷积核大小为 m，那么很容易得出卷积后的高度 h1 = h - m + 1。池化前的高度为 h1，池化滤波器大小为 s，那么池化后的高度为 h1 / s。对应到`model.summary()`的输出，输入大小为 (32, 32)，经过32个3x3的卷积核卷积后，大小为 (30, 30)，紧接着池化后，大小变为(15, 15)。

##### 3.2 全连接层

对图像进行分类，输出为一个长度为10的一维向量。通过`Dense`层，将3维卷积层输出转换为1维分类，即为全连接层。

首先通过：

```python
        tf.keras.layers.Flatten(),
```

将3维（R/G/B）图像深度（color_channels）数据展平为1维。

这里的深度指的是计算机用于产生颜色使用的信息。如果是黑白照片的话，高的单位就只有1；如果是彩色照片，就可能有红绿蓝三种颜色的信息，这时的深度为3。

##### 3.3 最终模型

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 64)                65600
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________
```

#### 4. 模型训练及可视化

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

history = model.fit(
    train_images, train_labels,
    epochs=100,
    batch_size=512,
    validation_split=0.2  # split 20% data as validation
)

hist = pd.DataFrame(history.history, index=history.epoch)
hist['loss'].plot(legend=True)
hist['val_loss'].plot(legend=True)
plt.show()
hist['accuracy'].plot(legend=True)
hist['val_accuracy'].plot(legend=True)
plt.show()
```

![](/assets/images/cifar10CNNLoss202004171754.png)





![](/assets/images/cifar10CNNAccuracy202004171754.png)

#### 5. 模型评价

##### 5.1 模型验证

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)
```

训练1,000次后，输出为：

```
1.4606272903442383 0.6808
```

##### 5.2 预测准确度

```python
preds = model.predict(test_images)
print(preds[0])
```

输出为：

```python
[3.0982800e-08 1.5912183e-11 1.5587782e-06 9.9738032e-01 7.4791600e-08 2.5771046e-03 3.9310959e-05 4.1453538e-09 1.6840111e-06 3.3820519e-11]
```

类似于“One-Hot”编码，每一列的值Value(i)表明了分类为i的概率。可以看到，第一张图片为“3”分类的概率最大9.9738032e-01。

利用Numpy的`argmax`返回最大值对应的索引，更直观展示。

```python
print(np.argmax(preds[0]))  # 3
print(test_labels[0])  # [3]
```

#### 6. 过拟合与欠拟合

根据训练中的Loss和Accuracy图，大概训练到40步后，模型开始出现过拟合。本例中，训练步数选择在40步即可。

#### References：

* [TensorFlow 2 中文文档 - 卷积神经网络分类 CIFAR-10](https://geektutu.com/post/tf2doc-cnn-cifar10.html)
* [The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR-10数据集说明](https://www.cnblogs.com/Jerry-Dong/p/8109938.html)
* [什么是卷积神经网络 CNN (Convolutional Neural Network)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-03-A-CNN/)

