from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.append('..')
from .backbone import EfficientDetBackbone
from losses.focalloss import FocalLoss
from .efficientdet.utils import BBoxTransform, ClipBoxes

class Detector(BaseModel):
    def __init__(self, model, n_classes, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.model = model
        self.model_name = "EfficientDet"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.set_optimizer_params()
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

        _, regression, classification, anchors = self.model(inputs)
        loss = self.criterion(classification, regression, anchors, labels)
        
        return loss

    
    def inference_step(self, batch, threshold = 0.2, iou_threshold=0.2):
        inputs = batch["imgs"]

        if self.device:
            inputs = inputs.to(self.device)

        _, regression, classification, anchors = self.model(inputs)
        regression = regression.detach()
        classification = classification.detach()
        anchors = anchors.detach()

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        outputs = self.model.detect(
            inputs, 
            anchors, 
            regression, 
            classification, 
            regressBoxes, 
            clipBoxes, 
            threshold = threshold, 
            iou_threshold=iou_threshold)

        return outputs

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

        _, regression, classification, anchors = self.model(inputs)
        loss = self.criterion(classification, regression, anchors, labels)

        metric_dict = {'map': 0}

    
        return loss , metric_dict

    def forward_test(self, size = 224):
        inputs = torch.rand(1,3,size,size)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    

    