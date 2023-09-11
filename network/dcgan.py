import numpy as np

from network import *
from custom.customlayers import Generator, Discriminator
import tensorflow as tf


class DCGAN(models.Model):
    def __init__(self,
                 **kwargs):
        super(DCGAN, self).__init__(**kwargs)
        self.g = Generator()
        self.d = Discriminator()

    def call(self, inputs, training=None, mask=None):
        random_noise,real_image = inputs
        fake_image = self.g(random_noise)
        score = self.d(fake_image)
        return fake_image, score


if __name__ == '__main__':
    a = tf.random.normal((4, 100))
    d = DCGAN()
    o = d(a)
