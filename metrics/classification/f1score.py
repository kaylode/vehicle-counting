import torch
import torch.nn as nn
import numpy as np

class F1ScoreMetric():
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, n_classes, average = 'macro'):
        self.n_classes = n_classes
        self.average = average
        self.reset()

    def update(self, outputs, targets):
        pred = torch.argmax(outputs, dim=1)
        pred = pred.cpu().numpy()
        targets = targets.cpu().numpy()
        for pd, gt in zip(pred, targets):
            self.count_dict[gt]['total_gt'] += 1
            self.count_dict[pd]['total_p'] += 1
            if pd == gt:
                self.count_dict[pd]['total_pt'] +=1
    
    def compute(self, item):
        try:
            precision = item['total_pt']*1.0 / item['total_p']
            recall = item['total_pt']*1.0 / item['total_gt']
        except ZeroDivisionError:
            return .0
        if precision+ recall >0:
            score = 2*precision*recall / (precision + recall)
        else:
            score = .0
        return score


    def compute_micro_average(self):
        total_p = sum([self.count_dict[i]['total_p'] for i in range(self.n_classes)])
        total_pt = sum([self.count_dict[i]['total_pt'] for i in range(self.n_classes)])
        total_gt = sum([self.count_dict[i]['total_gt'] for i in range(self.n_classes)])
        return self.compute({
            'total_p': total_p,
            'total_gt': total_gt,
            'total_pt': total_pt
        })

    def compute_macro_average(self):
        results = [self.compute(self.count_dict[i]) for i in range(self.n_classes)]
        score = sum(results) *1.0 / self.n_classes
        return score

    def value(self):
       
        if self.average == "micro":
            score = self.compute_micro_average()
        elif self.average =="macro":
            score = self.compute_macro_average()

        return {"f1-score" : score}

    def reset(self):
        self.count_dict = {
            i: {
                'total_gt': 0,
                'total_p' : 0,
                'total_pt': 0, 
            } for i in range(self.n_classes)
        }
    
    def __str__(self):
        return f'F1-Score: {self.value()}'

if __name__ == '__main__':
    f1 = F1ScoreMetric(3)
    y_true = torch.tensor([0, 1, 2, 0, 1, 2]).cuda()
    y_pred = torch.tensor([[3,1,2], [0,1,2], [1,3,1], [3,2,0], [3,1,1], [2,3,1]]).cuda()
    f1.update(y_pred ,y_true)
    print(f1.value())

