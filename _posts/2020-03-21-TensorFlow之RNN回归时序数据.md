---
title: TensorFlow之RNN回归时序数据及GRU函数使用
author: 赵旭山
tags: TensorFlow
typora-root-url: ..
---









```python
tf.keras.layers.GRU(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    recurrent_constraint=None, bias_constraint=None, dropout=0.0,
    recurrent_dropout=0.0, implementation=2, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, unroll=False,
    time_major=False, reset_after=True, **kwargs
)
```







#### References：

* [tf.keras.layers.GRU](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRU?hl=zh-cn)
* [Tensorflow 2.0 快速入门 —— RNN 预测牛奶产量](https://www.jianshu.com/p/e2ff67c7b7aa)
* [4_RNN_Many_to_One_TF2_0.ipynb](https://github.com/zht007/tensorflow-practice/blob/master/5_Prediction_MilkProdction/4_RNN_Many_to_One_TF2_0.ipynb)

