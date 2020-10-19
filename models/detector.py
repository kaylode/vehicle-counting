from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms


class Detector(BaseModel):
    def __init__(self, n_classes, **kwargs):
        super(Detector, self).__init__(**kwargs)
        """self.model = None
        self.model_name = "None"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
            self.criterion.to(self.device)"""
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            labels = [x.to(self.device) for x in labels]
        
        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        return loss

    
    def inference_step(self, batch):
        inputs = batch["imgs"]

        if self.device:
            inputs = inputs.to(self.device)

        loc_preds, cls_preds = self(inputs)
        
        #   TODO:
        # - add batch post-process

        outputs = self.model.detect(loc_preds, cls_preds)#[self.model.detect(i,j) for i,j in zip(loc_preds,cls_preds)]
            
        """if self.device:
            outputs['boxes'] = outputs['boxes'].cpu()
            outputs['labels'] = outputs['labels'].cpu()
            outputs['scores'] = outputs['scores'].cpu()"""
        
        return outputs

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            labels = [x.to(self.device) for x in labels]

        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        metric_dict = {'map': 0}
        """outputs = self.model.detect(
            loc_preds,
            cls_preds)

        metric_dict = self.update_metrics(
            outputs = {
                'det_boxes': outputs['boxes'],
                'det_labels': outputs['labels'],
                'det_scores': outputs['scores']},
            targets={
                'gt_boxes': boxes,
                'gt_labels': labels})"""
        
        return loss , metric_dict

    def forward_test(self, size = 224):
        inputs = torch.rand(1,3,size,size)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    

    