# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

from sklearn.metrics import confusion_matrix

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc










def eval_func(label_path, predict_path):
    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
            label = cv2.imread(lab_path,0)
            pre = cv2.imread(pre_path,0)
            label[label>0] = 1
            pre[pre>0] = 1
            label = np.uint8(label)
            pre = np.uint8(pre)
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts,labels)
    #print('acc: ',acc)
    #print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    #print('miou: ',miou)
    #print('fwavacc: ',fwavacc)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0


    pres = os.listdir(predict_path)
    
    for im in pres:
        lb_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)
        groundtruth = cv2.imread(lb_path,0)
        premask = cv2.imread(pre_path,0)
        seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
        true_pos += float(np.logical_and(premask, groundtruth).sum())  # float for division
        true_neg += np.logical_and(seg_inv, gt_inv).sum()
        false_pos += np.logical_and(premask, gt_inv).sum()
        false_neg += np.logical_and(seg_inv, groundtruth).sum()


    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)

    print('presion: ', prec)
    print('recall: ', rec)
    print('f1_score: ', F1)
    print('accuracy: ', accuracy)




def eval_new(label_list, pre_list):
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(pre_list, label_list)
    print('acc: ',acc)
    # print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    print('miou: ',miou)
    print('fwavacc: ',fwavacc)

    init = np.zeros((2,2))
    for i in range(len(label_list)):
        lab = label_list[i].flatten()
        pre = pre_list[i].flatten()
        confuse = confusion_matrix(lab, pre)
        init += confuse

    precision = init[1][1]/(init[0][1] + init[1][1])
    recall = init[1][1]/(init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1])/init.sum()
    f1_score = 2*precision*recall/(precision + recall)
    print('class_accuracy: ', precision)
    print('class_recall: ', recall)
    # print('accuracy: ', accuracy)
    print('f1_score: ', f1_score)


if __name__ == "__main__":
    label_path = './dataset/whubuild/test/labels/'
    predict_path = './dataset/whubuild/test/save/'
    label_path2 = './dataset/eval/GBE/'
    predict_path2 = './dataset/eval/BESNet/'

    eval_func(label_path, predict_path)
