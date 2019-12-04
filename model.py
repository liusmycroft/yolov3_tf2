import numpy as np
import tensorflow as tf


class Resblock_layer(tf.keras.layers.Layer):
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.run_time = num_blocks
        self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='valid',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            strides=(2, 2),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.batch = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.antivation = tf.keras.layers.PReLU()
        self.antivation2 = tf.keras.layers.PReLU()
        self.res_conv1 = tf.keras.layers.Conv2D(
            filters=num_filters // 2,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.res_conv2 = tf.keras.layers.Conv2D(
            filters=num_filters,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        print("resBlock is ini...")

    def call(self, inputs):
        # print("resBlock is running...")
        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.batch(x)
        x = self.antivation(x)
        for i in range(self.run_time):
            y = self.res_conv1(x)       # 重新根据新的x来进行卷积
            y = self.batch2(y)
            y = self.antivation2(y)
            y = self.res_conv2(y)
            y = self.batch(y)
            y = self.antivation(y)
            x = tf.keras.layers.Add()([x, y])
            
        return x


class Last_layers(tf.keras.layers.Layer):
    def __init__(self, num_filters, out_filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.conv1_2 = tf.keras.layers.Conv2D(
            filters=num_filters,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_filters * 2,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=out_filters,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.batch = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.antivation = tf.keras.layers.PReLU()
        self.antivation2 = tf.keras.layers.PReLU()
        self.antivation3 = tf.keras.layers.PReLU()
        
    
    def build(self, input_shape):
        print("last_layers is ini...")

    def call(self, inputs):
        tf.compat.v1.enable_eager_execution()
        x = self.conv1(inputs)
        x = self.batch(x)
        x = self.antivation(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.antivation2(x)
        x = self.conv1_2(x)
        x = self.batch3(x)
        x = self.antivation3(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.antivation2(x)
        x = self.conv1_2(x)
        x = self.batch3(x)
        x = self.antivation3(x)
        y = self.conv2(x)
        y = self.batch2(y)
        y = self.antivation2(y)
        y = self.conv3(y)

        return x, y


class Yolo(tf.keras.Model):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same)
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            kernel_initializer='glorot_uniform'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=256,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same)
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            kernel_initializer='glorot_uniform'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,             # 卷积层神经元（卷积核）数目
            kernel_size=[1, 1],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same)
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            kernel_initializer='glorot_uniform'
        )
        self.batch = tf.keras.layers.BatchNormalization()
        self.antivation = tf.keras.layers.PReLU()
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.antivation2 = tf.keras.layers.PReLU()
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.antivation3 = tf.keras.layers.PReLU()
        self.res1 = Resblock_layer(64, 1)
        self.res2 = Resblock_layer(128, 2)
        self.res3 = Resblock_layer(256, 8)
        self.res4 = Resblock_layer(512, 8)
        self.res5 = Resblock_layer(1024, 4)
        self.last1 = Last_layers(512, num_anchors*(num_classes + 5))
        self.last2 = Last_layers(256, num_anchors*(num_classes + 5))
        self.last3 = Last_layers(128, num_anchors*(num_classes + 5))
        self.upPooling = tf.keras.layers.UpSampling2D(2)
        self.connect = tf.keras.layers.Concatenate()
        self.upPooling2 = tf.keras.layers.UpSampling2D(2)
        self.connect2 = tf.keras.layers.Concatenate()

    def call(self, input_x):
        tf.compat.v1.enable_eager_execution()
        x = self.conv1(input_x)
        x = self.batch(x)
        x = self.antivation(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        res_52 = x
        x = self.res4(x)
        res_26 = x
        x = self.res5(x)
        x, y1 = self.last1(x)
        # print("==============================")
        # print("输出规格1")
        # print(tf.shape(y1))
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.antivation2(x)
        x = self.upPooling(x)
        x = self.connect([x, res_26])
        x, y2 = self.last2(x)
        # print("==============================")
        # print("输出规格2")
        # print(tf.shape(y2))
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.antivation3(x)
        x = self.upPooling2(x)
        x = self.connect2([x, res_52])
        x, y3 = self.last3(x)
        # print("==============================")
        # print("输出规格3")
        # print(tf.shape(y3))
        return [y1, y2, y3]
