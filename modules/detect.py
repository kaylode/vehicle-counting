import os
import numpy as np
import torch
from augmentations import TTA
from models import get_model, Detector
from trainer import get_class_names, load_checkpoint
from utils import postprocessing, download_pretrained_weights, CACHE_DIR



class ImageDetect:
    def __init__(self, args, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   
        self.debug = args.debug
        self.config = config
        self.min_iou = args.min_iou
        self.min_conf = args.min_conf
        self.max_dets=config.max_post_nms
        self.keep_ratio=config.keep_ratio
        self.fusion_mode=config.fusion_mode
        self.class_names = None
        self.mapping_dict = args.mapping     

        if args.tta:
            self.tta = TTA(
                min_conf=args.tta_conf_threshold, 
                min_iou=args.tta_iou_threshold, 
                postprocess_mode=args.tta_ensemble_mode)
        else:
            self.tta = None

        if args.weight is None:
            args.weight = os.path.join(CACHE_DIR, f'{config.model_name}.pth')
            download_pretrained_weights(f'{config.model_name}', args.weight)

        self.class_names, num_classes = get_class_names(args.weight)
        
        if self.mapping_dict is not None:
            self.included_classes = list(self.mapping_dict.keys())
            class_ids = list(self.mapping_dict.values())
            sorted_unique_class_ids = sorted(np.unique(class_ids))
            self.class_names = [self.class_names[i] for i in sorted_unique_class_ids]

        net = get_model(
            args, config,
            num_classes=num_classes)

        self.model = Detector(model = net, device = self.device)
        self.model.eval()
        
        if args.weight is not None:                
            load_checkpoint(self.model, args.weight)

        for param in self.model.parameters():
            param.requires_grad = False

    def run(self, batch):
        with torch.no_grad():
            boxes_result = []
            labels_result = []
            scores_result = []
                
            if self.tta is not None:
                preds = self.tta.make_tta_predictions(self.model, batch)
            else:
                preds = self.model.inference_step(batch)

            for idx, outputs in enumerate(preds):
                img_w = batch['image_ws'][idx]
                img_h = batch['image_hs'][idx]
                img_ori_ws = batch['image_ori_ws'][idx]
                img_ori_hs = batch['image_ori_hs'][idx]
                
                if self.mapping_dict is not None:
                    keep_idx = [enum for enum, i in enumerate(outputs["classes"]) if i-1 in self.included_classes]
                    labels = [self.mapping_dict[int(i)-1]+1 for i in outputs["classes"][keep_idx]]
                    outputs["classes"] = np.array(labels)
                    outputs['scores'] = outputs['scores'][keep_idx]
                    outputs['bboxes'] = outputs['bboxes'][keep_idx]

                outputs = postprocessing(
                    outputs, 
                    current_img_size=[img_w, img_h],
                    ori_img_size=[img_ori_ws, img_ori_hs],
                    min_iou=self.min_iou,
                    min_conf=self.min_conf,
                    max_dets=self.max_dets,
                    keep_ratio=self.keep_ratio,
                    output_format='xywh',
                    mode=self.fusion_mode)

                boxes = outputs['bboxes'] 

                # Here, labels start from 1, subtract 1 
                labels = outputs['classes'] - 1
                scores = outputs['scores']

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        return {
            "boxes": boxes_result, 
            "labels": labels_result,
            "scores": scores_result }