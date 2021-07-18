# Author: Zylo117
import os
import torch
import torchvision
import numpy as np
from torch import nn
from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, HeadNet
from .yolo import YoloLoss, Yolov4, non_max_suppression, Yolov5
from augmentations import MEAN, STD

def get_model(args, config, num_classes=None):
    global MEAN, STD
    max_post_nms = config.max_post_nms if config.max_post_nms > 0 else None
    max_pre_nms = config.max_pre_nms if config.max_pre_nms > 0 else None
    net = None

    if config.model_name.startswith('efficientdet'):
        compound_coef = config.model_name.split('-')[1]
        assert compound_coef in [f'd{i}' for i in range(8)], f"efficientdet version {compound_coef} is not supported"  
        net = EfficientDetBackbone(
            num_classes=num_classes, 
            compound_coef=compound_coef, 
            freeze_batchnorm = getattr(args, 'freeze_bn', False),
            max_pre_nms=max_pre_nms,
            image_size=config.image_size)

        # If use EfficientDet, use these numbers
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    elif config.model_name.startswith('yolo'):
        version_name = config.model_name.split('v')[1]
        net = YoloBackbone(
            version_name=version_name,
            num_classes=num_classes, 
            max_pre_nms=max_pre_nms,
            max_post_nms=max_post_nms)

        # If use YOLO, use these numbers
        MEAN = [0.0, 0.0, 0.0]
        STD = [1.0, 1.0, 1.0]
  
    # if args.sync_bn:
    #     net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)


    return net

class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass
    def forward(self, batch):
        pass
    def detect(self, batch):
        pass

class EfficientDetBackbone(BaseBackbone):
    def __init__(
        self, 
        num_classes=90, 
        compound_coef='d0', 
        image_size=[512,512], 
        freeze_batchnorm = False,
        max_pre_nms=None,
        max_post_nms=None,
        **kwargs):

        super(EfficientDetBackbone, self).__init__(**kwargs)
        self.name = f'efficientdet-{compound_coef}'
        config = get_efficientdet_config(f'tf_efficientdet_{compound_coef}')
        config.image_size = image_size
        config.norm_kwargs=dict(eps=.001, momentum=.01)

        if num_classes is None:
            num_classes = 90

        if max_pre_nms is None:
            max_pre_nms = 5000
        
        if max_post_nms is None:
            max_post_nms = 100

        config.max_detection_points = max_pre_nms 
        config.max_det_per_image = max_post_nms 
        
        net = EfficientDet(
            config, 
            pretrained_backbone=False)

        if freeze_batchnorm:
            print("freeze batchnorm")
            freeze_bn(net.backbone)
            freeze_bn(net.fpn)

        net.reset_head(num_classes=num_classes)
        net.class_net = HeadNet(config, num_outputs=num_classes)

        self.model = DetBenchTrain(net, config)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        targets = batch['targets']
        targets_res = {}

        inputs = inputs.to(device)
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        targets_res['boxes'] = boxes
        targets_res['labels'] = labels

        return self.model(inputs, targets_res)

    def detect(self, batch, device):
        inputs = batch["imgs"].to(device)
        img_sizes = batch['img_sizes'].to(device)
        img_scales = batch['img_scales'].to(device)

        outputs = self.model(inputs, inference=True, img_sizes=img_sizes, img_scales=img_scales)
        outputs = outputs.cpu().numpy()
        out = []
        for i, output in enumerate(outputs):
            boxes = output[:, :4]
            labels = output[:, -1]
            scores = output[:,-2]

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

class YoloBackbone(BaseBackbone):
    def __init__(
        self,
        version_name='5s',
        num_classes=None, 
        max_pre_nms=None,
        max_post_nms=None,
        **kwargs):

        super(YoloBackbone, self).__init__(**kwargs)

        if max_pre_nms is None:
            max_pre_nms = 30000
        self.max_pre_nms = max_pre_nms

        if max_post_nms is None:
            max_post_nms = 300
        self.max_post_nms = max_post_nms

        if num_classes is None:
            num_classes = 80

        version = version_name[0]
        if version=='4':
            version_mode = version_name.split('-')[1]
            self.name = f'yolov4-{version_mode}'
            self.model = Yolov4(
                cfg=f'./models/yolo/configs/yolov4-{version_mode}.yaml', ch=3, nc=num_classes
            )
        elif version =='5':
            version_mode = version_name[1]
            self.name = f'yolov5{version_mode}'
            self.model = Yolov5(
                cfg=f'./models/yolo/configs/yolov5{version_mode}.yaml', ch=3, nc=num_classes
            )

        self.model = nn.DataParallel(self.model).cuda()
        self.loss_fn = YoloLoss(
            num_classes=num_classes,
            model=self.model)

        self.num_classes = num_classes

    def forward(self, batch, device):
        inputs = batch["imgs"]
        targets = batch['yolo_targets']

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if self.model.training:
            outputs = self.model(inputs)
        else:
            _ , outputs = self.model(inputs)

        loss, loss_items = self.loss_fn(outputs, targets)

        ret_loss_dict = {
            'T': loss,
            'IOU': loss_items[0],
            'OBJ': loss_items[1],
            'CLS': loss_items[2],
        }
        return ret_loss_dict

    def detect(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs, _ = self.model(inputs)
        outputs = non_max_suppression(
            outputs, 
            conf_thres=0.0001, 
            iou_thres=0.8, 
            max_nms=self.max_pre_nms,
            max_det=self.max_post_nms) #[bs, max_det, 6]
    
        out = []
        for i, output in enumerate(outputs):
            # [x1,y1,x2,y2, score, label]
            if output is not None and len(output) != 0:
                output = output.detach().cpu().numpy()
                boxes = output[:, :4]
                boxes[:,[0,2]] = boxes[:,[0,2]] 
                boxes[:,[1,3]] = boxes[:,[1,3]] 

                # Convert labels to COCO format
                labels = output[:, -1] + 1
                scores = output[:, -2]
          
            else:
                boxes = []
                labels = []
                scores = []
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

def freeze_bn(model):
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if "BatchNorm2d" in classname:
            m.affine = False
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.eval() 
    model.apply(set_bn_eval)




    
