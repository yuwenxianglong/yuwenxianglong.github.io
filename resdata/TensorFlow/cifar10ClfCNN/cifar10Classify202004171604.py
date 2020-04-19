# -*- coding: utf-8 -*-
"""
@Project : cifar10ClfCNN
@Author  : Xu-Shan Zhao
@Filename: cifar10Classify202004171604.py
@IDE     : PyCharm
@Time1   : 2020-04-17 16:04:17
@Time2   : 2020/4/17 16:04
@Month1  : 4月
@Month2  : 四月
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images = train_images / 255.
test_images = test_images / 255.

plt.figure(figsize=(5, 3))
plt.subplots_adjust(hspace=0.1)
for n in range(15):
    plt.subplot(3, 5, n + 1)
    plt.imshow(train_images[n])
    plt.axis('off')
    plt.suptitle('CIFAR-10 Examples')  # plot total title

plt.show()

model = tf.keras.Sequential(
    [
        # 1st section
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 2nd section
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

checkpoint_path = 'train\cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, period=10
)

model.save(checkpoint_path.format(epoch=0))

history = model.fit(
    train_images, train_labels,
    epochs=100,
    batch_size=512,
    validation_split=0.2,
    callbacks=[cp_callback]
)

hist = pd.DataFrame(history.history, index=history.epoch)
hist['loss'].plot(legend=True)
hist['val_loss'].plot(legend=True)
plt.show()
hist['accuracy'].plot(legend=True)
hist['val_accuracy'].plot(legend=True)
plt.show()

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)

preds = model.predict(test_images)
print(preds[0])

print(np.argmax(preds[0]))
print(test_labels[0])

model.save('cifar10CNN.h5')

model = tf.keras.models.load_model('cifar10CNN.h5')
model.evaluate(train_images, train_labels)
model.evaluate(test_images, test_labels)
print(model.predict(test_images))