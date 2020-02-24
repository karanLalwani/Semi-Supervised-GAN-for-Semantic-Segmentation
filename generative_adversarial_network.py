# -*- coding: utf-8 -*-
"""
Created on Sun Fri 27 16:17:58 2019

@author: klal1
"""
from keras.layers import Lambda, Input
import keras.backend as K
from keras.models import Model
import tensorflow as tf

class GAN:
    def __init__(self, gen, dis, height, width):
        self.gen = gen
        self.dis = dis
        self.height = height
        self.width = width
        
    def supervisedCGAN(self):
        real_img = Input((self.height, self.width, 3))
        # seg_img = Input((height, width, NUM_CLASSES))
    
        generated_img = self.gen(real_img)
    
        generated_img_c = Lambda(lambda x:  K.argmax(x, axis=3), name='argmax', trainable=False)(generated_img)
        generated_img_c = Lambda(lambda x:  K.cast(x, dtype=tf.float32), name='cast', trainable=False)(generated_img_c)
        generated_img_c = Lambda(lambda x: K.expand_dims(x, axis=3), name='expand_dims', trainable=False)(generated_img_c)
        
        self.dis.trainable = False
        validity = self.dis([real_img, generated_img_c])
    
        return Model(inputs=[real_img], outputs=[validity, generated_img])
    
    def unsupervisedCGAN(self):
        real_img = Input((self.height, self.width, 3))
        # seg_img = Input((height, width, NUM_CLASSES))
    
        generated_img = self.gen(real_img)
    
        generated_img_c = Lambda(lambda x:  K.argmax(x, axis=3), name='argmax', trainable=False)(generated_img)
        generated_img_c = Lambda(lambda x:  K.cast(x, dtype=tf.float32), name='cast', trainable=False)(generated_img_c)
        generated_img_c = Lambda(lambda x: K.expand_dims(x, axis=3), name='expand_dims', trainable=False)(generated_img_c)
        
        self.dis.trainable = False
        validity = self.dis([real_img, generated_img_c])
    
        cgan = Model(inputs=[real_img], outputs=[validity])
        return cgan
        