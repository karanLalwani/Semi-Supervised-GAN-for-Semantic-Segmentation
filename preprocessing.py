# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:23:24 2019

@author: klal1
"""
import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
np.random.seed(123)

class Preprocessing:
    def __init__(self, height, width, n_classes):
        self.height = height
        self.width = width
        self.n_classes = n_classes
    
    def data_gen(self, images, imagePath, annotationPath, batch_size):
        while(True):
            X, y = [], []
            batch = np.random.choice(images, 2*batch_size)
            i = 0
            for image in batch:
                file = os.path.join(annotationPath, image+'.png')
                annotation = np.array(Image.open(file).resize((self.height, self.width))).reshape(self.height, self.width, 1)
                # annotation[annotation == 255] = 0
                if(any(x not in [0,1,2,3] for x in np.unique(annotation))):
                    continue
                y.append(to_categorical(annotation, 4))
                file = os.path.join(imagePath, image+'.jpg')
                X.append(np.array(Image.open(file).resize((self.height, self.width), Image.ANTIALIAS), 'float32').reshape(self.height, self.width,3)/255.0)
                i += 1
                if(i == batch_size):
                    break
            yield np.array(X), np.array(y)
            
    def get_test_train_filenames(self, imagePath, valPer, testPer):
        images = [image[:-4] for image  in os.listdir(imagePath)]
        l1 = int(len(images)*valPer)
        l2 = l1+int(len(images)*testPer)
        np.random.shuffle(images)
        testImages = images[:l1]
        valImages = images[l1:l2]
        trainImages = images[l2:]
        return trainImages, valImages, testImages
    
    def load_all_images(self, imagePath, annotationPath, images):
        X, y = [], []
        for fileName in images:
            file = os.path.join(annotationPath, fileName+'.png')
            annotation = np.array(Image.open(file).resize((self.height, self.width))).reshape(self.height, self.width, 1)
            # annotation[annotation == 255] = 0
            if(any(x not in [0,1,2,3] for x in np.unique(annotation))):
                continue
            y.append(to_categorical(annotation, self.n_classes))
            file = os.path.join(imagePath, fileName+'.jpg')
            X.append(np.array(Image.open(file).resize((self.height, self.width), Image.ANTIALIAS), 'float32').reshape(self.height,self.width,3)/255.0)
        return np.array(X), np.array(y)