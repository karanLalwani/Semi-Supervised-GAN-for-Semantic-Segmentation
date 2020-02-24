# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 00:44:05 2019

@author: klal1
"""
import numpy as np
from preprocessing import Preprocessing

class Metrics:
    def __init__(self, imagePath, segPath, fileNames, height, width, generator, labels):
        p = Preprocessing(height, width, len(labels))
        self.labels = labels
        self.X_data, self.y_data = p.load_all_images(imagePath, segPath, fileNames)
        self.y_predictions = generator.predict(self.X_data)
        self.cm = self.conf_matrix(np.argmax(self.y_data, axis=3), np.argmax(self.y_predictions, axis=3))
        self.dic = {}
        self.get_all_metrics()

    def conf_matrix(self, y_true, y_pred):
        conf_mat = np.zeros((len(self.labels), len(self.labels)))
        for b in range(y_true.shape[0]):
            for i in range(y_true.shape[1]):
                for j in range(y_true.shape[2]):
                    conf_mat[y_true[b][i][j]][y_pred[b][i][j]] += 1
        return np.array(conf_mat)
    
    def accuracy(self, cfMat):
        return np.sum(np.diag(cfMat))/cfMat.sum()
    
    def recall(self):
        r = np.diag(self.cm)/np.sum(self.cm, axis = 1)
        return np.mean(r), np.rec.fromarrays([self.labels.reshape(-1, 1), r.reshape(-1,1)], names=['label', 'value'])
    
    def precision(self):
        p = np.diag(self.cm)/np.sum(self.cm, axis = 0)
        return np.mean(p), np.rec.fromarrays([self.labels.reshape(-1, 1), p.reshape(-1,1)], names=['label', 'value'])
    
    def mIOU(self):
        iou = np.diag(self.cm)/(np.sum(self.cm, axis=0) + np.sum(self.cm, axis=1) - np.diag(self.cm))
        return np.mean(iou), np.rec.fromarrays([self.labels.reshape(-1, 1), iou.reshape(-1,1)], names=['label', 'value'])
    
    def f1_score(self, precision, recall):
        return (2*precision*recall)/(precision+recall)
    
    def specificity(self):
        pass
    
    def get_all_metrics(self):
        self.dic['full_accuracy'] = self.accuracy(self.cm)*100
        self.dic['class_accuracy'] = self.accuracy(self.cm[1:, 1:])*100
        p, perP = self.precision()
        self.dic['precision'], self.dic['per_class_precision'] = p, perP
        r, perR = self.recall()
        self.dic['recall'], self.dic['per_class_recall'] = r, perR
        self.dic['mIOU'], self.dic['IOU'] = self.mIOU()
        self.dic['f1_score'] = self.f1_score(p, r)
        self.dic['per_class_f1_score'] = self.f1_score(perP['value'], perR['value'])
    
    def printAllMetrics(self):
        print("class accuracy : ", self.dic['class_accuracy']); print("")
        print("full accuracy : ", self.dic['full_accuracy']); print("")
        print("Intersection over Union : ", self.dic['IOU']); print("")
        print("mean Intersection over Union\n", self.dic['mIOU']); print("")
        print("recall : ", self.dic['recall']); print("")
        print("per_class_recall\n", self.dic['per_class_recall']); print("")
        print("precision : ", self.dic['precision']); print("")
        print("per_class_percision\n", self.dic['per_class_precision']); print("")
        print("f1_score\n", self.dic['f1_score']); print("")
        print("per_class_f1_score\n", self.dic['per_class_f1_score']); print("")