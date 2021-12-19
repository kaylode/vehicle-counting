import torch
import numpy as np
from torch import nn
from augmentations import MEAN, STD

def get_model(args, config):
    global MEAN, STD

    net = YoloBackbone(
        weight=args.weight,
        min_iou=config.min_iou,
        min_conf=config.min_conf,
        max_det=config.max_det,
        filter_classes=args.mapping_dict.keys())

        
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


        print('es')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, force_reload=True) 

        self.class_names = self.model.names
        
        self.model.conf = min_conf  # NMS confidence threshold
        self.model.iou = min_iou  # NMS IoU threshold
        self.model.classes = filter_classes   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image

    def detect(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        results = self.model(inputs)  # inference
        outputs = results.pandas().xyxy
    
        out = []
        for i, output in enumerate(outputs):            
            output = output.to_json(orient="records")

            boxes = []
            labels = []
            scores = []

            for obj in output:
                boxes.append([obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax']])
                labels.append(obj["class"])
                scores.append(obj["confidence"])
          
            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out




    
