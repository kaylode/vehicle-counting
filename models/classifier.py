from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms



class Classifier(BaseModel):
    def __init__(self, backbone, n_classes, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.model = backbone
        self.model_name = "Classifier"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.set_optimizer_params()
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False
  

        if self.device:
            self.model.to(self.device)
            self.criterion.to(self.device)
    

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["labels"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    
    def inference_step(self, batch):
        inputs = batch['imgs']
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)

        if self.device:
            preds = preds.cpu()
        return preds.numpy()

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["labels"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        outputs = self(inputs) #batchsize, label_dim
        loss = self.criterion(outputs, targets)

        metric_dict = self.update_metrics(outputs, targets)
        
        return loss , metric_dict

    def forward_test(self):
        inputs = torch.rand(1,3,224,224)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    

    