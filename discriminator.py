# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:19:22 2019

@author: klal1
"""

from keras.layers import Conv2D, Input, Concatenate, Activation, LeakyReLU
from keras.models import Model

class Discriminator:
    def __init__(self, img1_shape, img2_shape):
        self.img1_shape, self.img2_shape = img1_shape, img2_shape

    def CFCDiscriminator(self):
        img_A = Input(self.img1_shape)
        img_B = Input(self.img2_shape)
        
        # lay_1_A = Conv2D(3, 3, padding='same', name='block1_conv_A')(img_A)    
        # lay_1_B = Conv2D(3, 3, padding='same', name='block1_conv_B')(img_B)
        con_1 = Concatenate(name='concat_1')([img_A, img_B])
    
        lay_1 = Conv2D(64, 4, strides=2, padding='same', name='block1_conv')(con_1)
        # lay_2 = BatchNormalization(name='block2_batchNorm')(lay_2)
        lay_1 = LeakyReLU(alpha=0.2)(lay_1)
    
        lay_2 = Conv2D(128, 4, strides=2, padding='same', name='block2_conv')(lay_1)
        # lay_2 = BatchNormalization(name='block2_batchNorm')(lay_2)
        lay_2 = LeakyReLU(alpha=0.2)(lay_2)
    
        lay_3 = Conv2D(256, 4, strides=2, padding='same', name='block3_conv')(lay_2)
        # lay_3 = BatchNormalization(name='block3_batchNorm')(lay_3)
        lay_3 = LeakyReLU(alpha=0.2)(lay_3)
    
        lay_4 = Conv2D(256, 4, strides=1, padding='same', name='block4_conv')(lay_3)
        # lay_4 = BatchNormalization(name='block4_batchNorm')(lay_4)
        lay_4 = LeakyReLU(alpha=0.2)(lay_4)    
    
        lay_5 = Conv2D(512, 4, strides=1, padding='same', name='block5_conv')(lay_4)
        # lay_5 = BatchNormalization(name='block5_batchNorm')(lay_5)
        lay_5 = LeakyReLU(alpha=0.2)(lay_5)
    
        # lay_6 = Conv2D(512, 3, padding='same', name='block6_conv')(lay_5)
        # lay_6 = BatchNormalization(name='block6_batchNorm')(lay_6)
        # lay_6 = LeakyReLU(alpha=0.2)(lay_6)
    
        # lay_7 = Conv2D(512, 3, padding='same', name='block7_conv')(lay_6)
        # lay_7 = BatchNormalization(name='block7_batchNorm')(lay_7)
        # lay_7 = LeakyReLU(alpha=0.2)(lay_7)
    
        # lay_8 = Conv2D(1024, 3, padding='same', name='block8_conv')(lay_7)
        # lay_8 = BatchNormalization(name='block8_batchNorm')(lay_8)
        # lay_8 = LeakyReLU(alpha=0.2)(lay_8)
    
        # lay_7 = Conv2D(1, 1, padding='same', name='block6_conv')(lay_5)
        # lay_8 = UpSampling2D(4)(lay_7)
    
        # lay_9 = Conv2D(1, 1, padding='same', name='block7_conv')(lay_8)
        # lay_10 = UpSampling2D(4)(lay_9)
    
        lay_6 = Conv2D(1, 4, padding='same', name='block8_conv')(lay_5)
        validity = Activation('sigmoid', name='binary_output')(lay_6)
        
        return Model([img_A, img_B], validity)