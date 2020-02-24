# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:15:27 2019

@author: klal1
"""

from keras.layers import Conv2D, DepthwiseConv2D, Input, Concatenate, Add, UpSampling2D
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, Reshape
from keras.models import Model

#from keras.applications.xception import Xception

class Generator:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        

    def separable_conv(self, x, filters, kernel_size, strides=(1,1), padding='valid', dilation_rate=(1, 1), activation=None, name=None):
        
        x = DepthwiseConv2D(kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding, dilation_rate=dilation_rate, name=name)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        return x
    
    def deepLab(self, type='custom'):
        
        if(type=='pretrained'):
            pass
        
        inp_img = Input(shape = self.input_shape, name='InputImage')
    
        # xcept = Xception(include_top=False, weights='imagenet', input_tensor=inp_img, input_shape=input_shape, pooling=None)
        # x_14 = xcept.output
        
        x_1 = Conv2D(32, 3, strides=2, padding='same', activation='relu', use_bias=True, name='entry_1_conv')(inp_img)
        x_2 = Conv2D(64, 3, padding='same', activation='relu', use_bias=True, name='entry_2_conv')(x_1)
        x_2 = BatchNormalization()(x_2)
        
        ##### Entry Flow #####
        
        x_3 = self.separable_conv(x_2, 128, 3, padding='same', activation='relu', name='entry_3_1_SepConv')
        x_3 = self.separable_conv(x_3, 128, 3, padding='same', activation='relu', name='entry_3_2_SepConv')
        x_3 = self.separable_conv(x_3, 128, 3, strides=2, padding='same', activation='relu', name='entry_3_3_SepConv')
        x_t = Conv2D(128, 1, strides=2, padding='same')(x_2)
        x_3 = Add()([x_t, x_3])
        
        x_4 = self.separable_conv(x_3, 256, 3, padding='same', activation='relu', name='entry_4_1_SepConv')
        x_4 = self.separable_conv(x_4, 256, 3, padding='same', activation='relu', name='entry_4_2_SepConv')
        x_4 = self.separable_conv(x_4, 256, 3, strides=2, padding='same', activation='relu', name='entry_4_3_SepConv')
        x_t = Conv2D(256, 1, strides=2, padding='same')(x_3)
        x_4 = Add()([x_t, x_4])
        
        x_5 = self.separable_conv(x_4, 728, 3, padding='same', activation='relu', name='entry_5_1_SepConv')
        x_5 = self.separable_conv(x_5, 728, 3, padding='same', activation='relu', name='entry_5_2_SepConv')
        x_5 = self.separable_conv(x_5, 728, 3, strides=2, padding='same', activation='relu', name='entry_5_3_SepConv')
        x_t = Conv2D(728, 1, strides=2, padding='same')(x_4)
        x_5 = Add()([x_t, x_5])
        
        
        ##### Middle Flow #####
        
        _x = x_5
        for i in range(16):
            x = self.separable_conv(_x, 728, 3, padding='same', activation='relu')
            x = self.separable_conv(x, 728, 3, padding='same', activation='relu')
            x = self.separable_conv(x, 728, 3, padding='same', activation='relu')
            _x = Add()([_x, x])
        
        x_13 = self.separable_conv(_x, 728, 3, padding='same', activation='relu', name='exit_13_1_SepConv')
        x_13 = self.separable_conv(x_13, 1024, 3, padding='same', activation='relu', name='exit_13_2_SepConv')
        x_13 = self.separable_conv(x_13, 1024, 3, padding='same', activation='relu', name='exit_13_3_SepConv')
        x_t = Conv2D(1024, 1, padding='same')(_x)
        x_13 = Add()([x_t, x_13])
        
        x_14 = self.separable_conv(x_13, 1536, 3, padding='same', activation='relu')
        x_14 = self.separable_conv(x_14, 1536, 3, padding='same', activation='relu')
        x_14 = self.separable_conv(x_14, 2048, 3, padding='same', activation='relu')
    
    
        ##### Atrous spatial pyramid pooling #####
    
        x_p1 = GlobalAveragePooling2D()(x_14)
        x_p1 = Reshape((1, 1, -1))(x_p1)
        x_p1 = Conv2D(256, 1, padding='same')(x_p1)
        x_p1 = UpSampling2D(16)(x_p1)
        
        x_a1 = self.separable_conv(x_14, 256, 1, padding='same', activation='relu')
        
        x_a2 = self.separable_conv(x_14, 256, 3, padding='same', dilation_rate=6, activation='relu')
        
        x_a3 = self.separable_conv(x_14, 256, 3, padding='same', dilation_rate=12, activation='relu')
        
        x_a4 = self.separable_conv(x_14, 256, 3, padding='same', dilation_rate=18, activation='relu')
        
        ##### Concatination #####
        
        con = Concatenate(name='atrous_spatial_pyramid_concat')([x_a1, x_a2, x_a3, x_a4, x_p1])
        x_15 = Conv2D(48, 1, padding='same', activation='relu')(con)
        
        x_16 = UpSampling2D(4)(x_15)
        # x_t = Conv2D(48, 1)(xcept.get_layer('conv2d_2').output)
        x_t = Conv2D(48, 1, padding='same')(x_3)
        x_17 = Concatenate(name='lay4_upsample4_concat')([x_t, x_16])
        
        x_18 = Conv2D(24, 3, padding='same', activation='relu')(x_17)
        x_19 = UpSampling2D(4, name='lay5_upsample4_concat')(x_18)
    
        x_20 = Conv2D(self.num_classes, 3, padding='same', activation='softmax')(x_19)
        # x_21 = UpSampling2D(2, name='categorical_output')(x_20)
        
        
        return Model(inp_img, x_20)