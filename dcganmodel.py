import math
import random
from custom.customlayers import Generator, Discriminator
import numpy as np
from network.dcgan import DCGAN
import tensorflow as tf
import tensorflow.keras as keras
import cv2
from PIL import Image


class DCGANModlel():
    def __init__(self,
                 lr: float,
                 **kwargs):
        super(DCGANModlel, self).__init__(**kwargs)

        self.network = DCGAN()

        # 对判别器和生成器使用不同的学习速度。使用较低的学习率更新生成器，判别器使用较高的学习率进行更新。
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
        self.optimizer_d = keras.optimizers.Adam(learning_rate=lr * 3, beta_1=0.0)
        self.optimizer_g = keras.optimizers.Adam(learning_rate=lr, beta_1=0.0)
        self.train_loss_d = keras.metrics.Mean()
        self.train_loss_g = keras.metrics.Mean()

    # 异步训练GAN行不通
    # def train(self, random_noise, real_image):
    #     # 一个mini-batch中必须只有real数据或者fake数据，不要把他们混在一起训练。
    #     with tf.GradientTape() as tape_d:
    #         fake_images = self.network.g(random_noise)
    #         real_score = self.network.d(real_image)
    #         real_loss = self.loss_fn(y_true=tf.ones_like(real_score), y_pred=real_score)
    #         fake_score = self.network.d(fake_images)
    #         fake_loss = self.loss_fn(y_true=tf.zeros_like(fake_score), y_pred=fake_score)
    #         loss_d = tf.concat([real_loss, fake_loss], axis=-1)
    #     gradient_d = tape_d.gradient(loss_d, self.network.d.trainable_variables)
    #     self.optimizer_d.apply_gradients(zip(gradient_d, self.network.d.trainable_variables))
    #     with tf.GradientTape() as tape_g:
    #         fake_image_srcs = self.network.g(random_noise)
    #         score = self.network.d(fake_image_srcs)
    #         loss_g = self.loss_fn(y_true=tf.ones_like(score), y_pred=score)
    #     gradient_g = tape_g.gradient(loss_g, self.network.g.trainable_variables)
    #     self.optimizer_g.apply_gradients(zip(gradient_g, self.network.g.trainable_variables))
    #
    #     self.train_loss_d(loss_d)
    #     self.train_loss_g(loss_g)

    def train_step(self, random_noise, real_images):
        # 在同一个tape中同步更新discriminator和generator，计算速度会变快
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            fake_images = self.network.g(random_noise)

            real_score = self.network.d(real_images)
            fake_score = self.network.d(fake_images)

            real_loss = self.loss_fn(tf.ones_like(real_score), real_score)
            fake_loss = self.loss_fn(tf.zeros_like(fake_score), fake_score)
            # d_loss = real_loss + fake_loss
            d_loss = tf.concat([real_loss, fake_loss], axis=-1)
            g_loss = self.loss_fn(tf.ones_like(fake_score), fake_score)
        gradients_d = tape_d.gradient(d_loss, self.network.d.trainable_variables)
        gradients_g = tape_g.gradient(g_loss, self.network.g.trainable_variables)
        self.optimizer_d.apply_gradients(zip(gradients_d, self.network.d.trainable_variables))
        self.optimizer_g.apply_gradients(zip(gradients_g, self.network.g.trainable_variables))
        self.train_loss_d(d_loss)
        self.train_loss_g(g_loss)

    @staticmethod
    def merge(fake_images, gap=4):
        fake_images = (fake_images + 1) * 127.5
        fake_images = np.array(fake_images, dtype=np.uint8)
        fake_images = np.clip(fake_images, 0, 255)
        fake_images = [Image.fromarray(img) for img in fake_images]
        w, h = fake_images[0].size
        newimg_len = int(math.sqrt(len(fake_images)))
        newimg = Image.new(fake_images[0].mode,
                           size=((w + gap) * newimg_len - gap, (h + gap) * newimg_len - gap),
                           color=(255, 255, 255))
        i = 0
        for row in range(newimg_len):
            for col in range(newimg_len):
                newimg.paste(fake_images[i], box=((w + gap) * row, (h + gap) * col))
                i = i + 1
        return np.array(newimg)

    def fake_image_save(self, path, num_imgs, epoch=0):
        assert num_imgs > -1 and (num_imgs ** 0.5 % 1 == 0), 'num_imgs必须是被开方数为整数'
        random_noise = tf.random.normal(shape=(num_imgs, 100))
        fake_images = self.network.g(random_noise)
        newimg = self.merge(fake_images)
        cv2.imwrite(path + f'\\epoch{epoch + 1}.jpg', newimg)


if __name__ == '__main__':
    from custom.customlayers import Generator
    import configure.config as cfg

    g = Generator()
    random_noise = tf.random.normal(shape=(25, 1, 1, 100))
    fake_images = g(random_noise)
    model = DCGANModlel(lr=1e-3)
    model.fake_image_save(path=cfg.result_path, num_imgs=25)
