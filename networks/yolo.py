import os
import json
import torch
import numpy as np
from torch import nn
from augmentations import MEAN, STD
from utilities.utils import download_pretrained_weights

CACHE_DIR = './.cache'

def get_model(args, config):
    global MEAN, STD

    if args.weight is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        args.weight = os.path.join(CACHE_DIR, f'{config.model_name}.pt')
        download_pretrained_weights(f'{config.model_name}', args.weight)


    filter_classes = None if not args.mapping_dict else args.mapping_dict.keys()

    net = YoloBackbone(
        weight=args.weight,
        min_iou=config.min_iou,
        min_conf=config.min_conf,
        max_det=config.max_det,
        filter_classes=filter_classes)

        
    # If use YOLO, use these numbers
    MEAN = [0.0, 0.0, 0.0]
    STD = [1.0, 1.0, 1.0]
  
    return net

class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass
    def forward(self, batch):
        pass
    def detect(self, batch):
        pass

class YoloBackbone(BaseBackbone):
    def __init__(
        self,
        weight,
        min_iou,
        min_conf,
        max_det,
        filter_classes=None,
        **kwargs):

        super().__init__(**kwargs)


        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, force_reload=True) 

        self.class_names = self.model.names
        
        self.model.conf = min_conf  # NMS confidence threshold
        self.model.iou = min_iou  # NMS IoU threshold
        self.model.classes = list(filter_classes) if filter_classes is not None else None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image

    def detect(self, batch, device):
        inputs = batch["imgs"]
        results = self.model(inputs)  # inference
        
        outputs = results.pandas().xyxy
    
        out = []
        for i, output in enumerate(outputs):            
            output = json.loads(output.to_json(orient="records"))

            boxes = []
            labels = []
            scores = []
            for obj_dict in output:
                boxes.append([obj_dict['xmin'], obj_dict['ymin'], obj_dict['xmax']-obj_dict['xmin'], obj_dict['ymax']-obj_dict['ymin']])
                labels.append(obj_dict["class"])
                scores.append(obj_dict["confidence"])
          
            if len(boxes) > 0:
                out.append({
                    'bboxes': np.array(boxes),
                    'classes': np.array(labels),
                    'scores': np.array(scores),
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out




    
