# !usr/bin/python

# -*- coding: utf-8 -*-

from InceptionModel import InceptionModel
import sys
import os
import numpy as np
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod,DeepFool
from cleverhans.model import Model
slim = tf.contrib.slim

tensorflow_master = ""
inputimg="./1.jpg"
checkpoint_path = "./inception_v3.ckpt"
output_dir = "./outcarplate"
max_epsilon = 4.0
image_width = 299
image_height = 299
batch_size = 50
sys.path.append('./cleverhans')
eps = 2* max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
nb_classes = 1001
print(-1)
class attack(InceptionModel):
    def load_images(self,inputimg, batch_shape):
        images = np.zeros(batch_shape)
        filenames = []
        idx = 0
        batch_size = batch_shape[0]
        with tf.io.gfile.GFile(inputimg, "rb") as f:
            images[idx, :, :, :] = imread(f).astype(np.float) * 2.0 / 255.0 - 1.0
        filenames.append(os.path.basename(inputimg))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
        if idx > 0:
            yield filenames, images

    def save_images(self,images, filenames, output_dir):
        for i, filename in enumerate(filenames):
            # Images for inception classifier are normalized to be in [-1, 1] interval,
            # so rescale them back to [0, 1].
            with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
                img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
                Image.fromarray(img).save(f, format='JPEG')
    def run(self,inputimg):
        print(1)
        # 加载图片
        image_iterator = self.load_images(inputimg, batch_shape)
        # 得到第一个batch的图片
        filenames, images = next(image_iterator)
        # 日志
        #tf.logging.set_verbosity(tf.logging.INFO)
        print(2)
        with tf.Graph().as_default():
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
            # 实例一个model
            # 开启一个会话
            ses = tf.Session()
            # 对抗攻击开始
            model =InceptionModel(nb_classes)
            fgsm =FastGradientMethod(model)
            x_adv = fgsm.generate(x_input, eps=eps, ord=np.inf, clip_min=-1., clip_max=1.)
            # fgsm_a是基于L2范数生成的对抗样本
            # fgsm_a = FastGradientMethod(model)
            # x_adv = fgsm_a.generate(x_input, ord=2, clip_min=-1., clip_max=1.)
            # 恢复inception-v3 模型
            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                 scaffold=tf.train.Scaffold(saver=saver),
                 checkpoint_filename_with_path=checkpoint_path,
                 master=tensorflow_master)
            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                nontargeted_images = sess.run(x_adv, feed_dict={x_input: images})
                self.save_images(nontargeted_images, filenames, output_dir)
            path="".join(filenames)
            return(output_dir+"/"+path)
        print(3)

if __name__ == '__main__':
    a=InceptionModel(Model)
    A=attack(a)
    A.run(inputimg)