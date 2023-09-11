from custom import *
import tensorflow as tf


class GConvBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple = (5, 5),
                 strides: tuple = (2, 2),
                 padding: str = 'SAME',
                 activation: str = 'relu',
                 use_bn: bool = True,
                 **kwargs):
        super(GConvBlock, self).__init__(**kwargs)
        self.use_bn = use_bn

        self.conv = layers.Conv2DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = layers.BatchNormalization()
        self.activation = activations.get(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class Generator(layers.Layer):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.convblock_1 = GConvBlock(filters=512, strides=(4, 4))
        self.convblock_2 = GConvBlock(filters=256)
        self.convblock_3 = GConvBlock(filters=128)
        self.convblock_4 = GConvBlock(filters=64)
        self.convblock_5 = GConvBlock(filters=3, activation='tanh', use_bn=False)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 1, 1, -1))
        x = layers.BatchNormalization()(inputs)
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.convblock_5(x)
        return x


class DConvBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple = (5, 5),
                 strides: tuple = (2, 2),
                 padding: str = 'SAME',
                 activation: str = 'relu',
                 use_bn: bool = True,
                 **kwargs):
        super(DConvBlock, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.activation = activation

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)
        self.bn = layers.BatchNormalization()
        self.activation = activations.get(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        if self.activation == 'relu':
            x = self.activation(x, alpha=0.2)
        else:
            x = self.activation(x)
        return x


class Discriminator(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.convblock_1 = DConvBlock(filters=64)
        self.convblock_2 = DConvBlock(filters=128)
        self.convblock_3 = DConvBlock(filters=256)
        self.convblock_4 = DConvBlock(filters=512)
        self.convblock_5 = DConvBlock(filters=1, kernel_size=(4, 4), padding='valid', use_bn=False,
                                      activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        x = self.convblock_1(inputs)
        x = self.convblock_2(x)
        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.convblock_5(x)
        x = tf.squeeze(x,axis=[1,2])
        return x
if __name__ == '__main__':
    import numpy as np
    g = Generator()
    o1 = g(np.random.normal(size=(4,100)))
    w1=g.get_weights()
    d = Discriminator()
    o2 = d(np.random.normal(size=(4,64,64,3)))
    w2 = d.get_weights()
    1