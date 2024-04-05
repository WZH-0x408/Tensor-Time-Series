from ..benchmark_manager import EvaluatorBase
import numpy as np

class ConfuseMatrix(EvaluatorBase):
    def __init__(self) -> None:
        pass

    def eval(self, pred, gt):
        pred = np.array(pred)
        gt = np.array(gt)
        if len(pred.shape) == 2:
            pred = np.argmax(pred, axis=1)
        if len(gt.shape) == 2:
            gt = np.argmax(gt, axis=1)
        self.num_classes = np.unique(np.concatenate((pred, gt))).size
        self.all = len(pred)
        self.TP, self.TN, self.FP, self.FN = self.cal_confuse_matrix(pred, gt)
        self.acc = self.cal_accuracy()
        self.pre = self.cal_precision()
        self.recall = self.cal_recall()
        self.f1 = self.cal_f1()
        self.macro_f1 = self.cal_macro_f1()
        # show resulst
        print(">> ClassificationEvaluator:")
        print("\tC\tTP\tTN\tFP\tFN\tAcc\tPre\tRecall\tF1")
        for i in range(self.num_classes):
            print(f'\t{i}\t{self.TP[i]}\t{self.TN[i]}\t{self.FP[i]}\t{self.FN[i]}\t{self.acc[i]:.3f}\t{self.pre[i]:.3f}\t{self.recall[i]:.3f}\t{self.f1[i]:.3f}')
        print(f"\tMacro-F1: {self.macro_f1:.3f}")

    def cal_confuse_matrix(self, pred, gt):
        TP = np.zeros(self.num_classes, dtype=np.int16)
        TN = np.zeros(self.num_classes, dtype=np.int16)
        FP = np.zeros(self.num_classes, dtype=np.int16)
        FN = np.zeros(self.num_classes, dtype=np.int16)
        for i in range(self.num_classes):
            TP[i] = np.sum((pred == i) & (gt == i))
            TN[i] = np.sum((pred != i) & (gt != i))
            FP[i] = np.sum((pred == i) & (gt != i))
            FN[i] = np.sum((pred != i) & (gt == i))
        return TP, TN, FP, FN
    
    def cal_accuracy(self):
        accuracy = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            accuracy[i] = (self.TP[i]+self.TN[i]) / (self.all)
        return accuracy

    def cal_precision(self):
        precision = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            precision[i] = (self.TP[i]) / (self.TP[i]+self.FP[i])
        return precision

    def cal_recall(self):
        recall = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            recall[i] = (self.TP[i]) / (self.TP[i]+self.FN[i])
        return recall
    
    def cal_f1(self):
        f1_score  = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            f1_score[i] = 2*(self.pre[i] * self.recall[i])/(self.pre[i] + self.recall[i])
        return f1_score
    
    def cal_macro_f1(self):
        macro_f1 = np.sum(self.f1) / len(self.f1)
        return macro_f1