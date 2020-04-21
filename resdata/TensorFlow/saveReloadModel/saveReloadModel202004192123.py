# -*- coding: utf-8 -*-
"""
@Project : saveReloadModel
@Author  : Xu-Shan Zhao
@Filename: saveReloadModel202004192123.py
@IDE     : PyCharm
@Time1   : 2020-04-19 21:23:59
@Time2   : 2020/4/19 21:23
@Month1  : 4月
@Month2  : 四月
"""

import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255., test_images / 255.


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

checkpoint_path = 'train\cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save(checkpoint_path.format(epoch=0))

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

hist = pd.DataFrame(history.history, index=history.epoch)
hist['loss'].plot(legend=True)
hist['val_loss'].plot(legend=True)
plt.show()
hist['accuracy'].plot(legend=True)
hist['val_accuracy'].plot(legend=True)
plt.show()

latest = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(latest)
# model = create_model()
# model.load_weights(latest)
print('\nLoad model from latest checkpoint.')
model_fromCP = tf.keras.models.load_model(latest)
model_fromCP.evaluate(train_images, train_labels)

print('\nLoad model from fully saved HDF5 file.')
model.save('cifar10CNN.h5')
model_fromHDF5 = tf.keras.models.load_model('cifar10CNN.h5')
model_fromHDF5.evaluate(train_images, train_labels)
