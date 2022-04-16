tf.layers.dense(

    inputs,  # 输入该网络层的数据

    units,  # 输出的维度大小，改变inputs的最后一维

    activation=None,  # 激活函数，即神经网络的非线性变化

    use_bias=True,  # 使用bias为True（默认使用），不用bias改成False即可，是否使用偏置项

    kernel_initializer=None,  # 卷积核的初始化器

    bias_initializer=tf.zeros_initializer(),  # 偏置项的初始化器，默认初始化为0

    kernel_regularizer=None,  # 卷积核的正则化，可选

    bias_regularizer=None,  # 偏置项的正则化，可选

    activity_regularizer=None,  # 输出的正则化函数

    kernel_constraint=None,

    bias_constraint=None,

    trainable=True,  # 表明该层的参数是否参与训练。如果为真则变量加入到图集合中

    name=None,  # 层的名字

    reuse=None  # 是否重复使用参数

)
