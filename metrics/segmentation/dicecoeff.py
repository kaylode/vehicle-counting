import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

class DiceScore():
    def __init__(self, num_classes, ignore_index = None, eps=1e-6, thresh = 0.5):
        self.thresh = thresh
        self.num_classes = num_classes
        self.pred_type = "multi" if num_classes > 1 else "binary"

        if num_classes == 1:
            self.num_classes+=1
        
        self.ignore_index = ignore_index
        self.eps = eps

        self.scores_list = np.zeros(self.num_classes)
        self.reset()

    def compute(self, outputs, targets): 
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)
      
        batch_size, _ , w, h = outputs.shape
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
      
        one_hot_targets = torch.zeros(batch_size, self.num_classes, h, w)
        one_hot_predicts = torch.zeros(batch_size, self.num_classes, h, w)
        
        if self.pred_type == 'binary':
            predicts = (outputs > self.thresh).float()
        elif self.pred_type =='multi':
            predicts = torch.argmax(outputs, dim=1).unsqueeze(1)

        one_hot_targets.scatter_(1, targets.long(), 1)
        one_hot_predicts.scatter_(1, predicts.long(), 1)
        
        for cl in range(self.num_classes):
            cl_pred = one_hot_predicts[:,cl,:,:]
            cl_target = one_hot_targets[:,cl,:,:]
            score = self.binary_compute(cl_pred, cl_target)
            self.scores_list[cl] += sum(score)
        

    def binary_compute(self, predict, target):
        # outputs: (batch, 1, W, H)
        # targets: (batch, 1, W, H)

        intersect = (predict * target).sum((-2,-1))
        union = (predict + target).sum((-2,-1))
        return (2. * intersect + self.eps) / (union +self.eps)
        
    def reset(self):
        self.scores_list = np.zeros(self.num_classes)
        self.sample_size = 0

    def update(self, outputs, targets):
        self.sample_size += outputs.shape[0]
        self.compute(outputs, targets)

    def value(self):
        scores_each_class = self.scores_list / self.sample_size #mean over number of samples
        if self.pred_type == 'binary':
            scores = scores_each_class[1] # ignore background which is label 0
        else:
            scores = sum(scores_each_class) / self.num_classes
        return {"dice_score" : np.round(scores, decimals=4)}

    def summary(self):
        class_iou = self.scores_list / self.sample_size #mean
        
        print(f'{self.value()}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i}: {x:.4f}')

    def __str__(self):
        return f'Dice Score: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

    

    