# -*- coding: utf-8 -*-
"""
@Project : cifar10ClfCNN202004142235
@Author  : Xu-Shan Zhao
@Filename: cifar10CLFCNN202004142248.py
@IDE     : PyCharm
@Time1   : 2020-04-14 22:48:27
@Time2   : 2020/4/14 10:48 下午
@Month1  : 4月
@Month2  : 四月
"""

import matplotlib.pyplot as plt
import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

plt.figure(figsize=(4, 3))
plt.subplots_adjust(hspace=0.1)
for n in range(15):
    plt.subplot(3, 5, n+1)
    plt.imshow(train_x[n])
    plt.axis('off')

_ = plt.suptitle("First 15th samples of CIFAR-10")
plt.show()