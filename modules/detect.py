import os
import numpy as np
import torch
from networks import get_model, Detector



class ImageDetect:
    def __init__(self, args, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   
    
        self.mapping_dict = args.mapping     

        net = get_model(args, config)
        self.class_names = net.class_names
        
        if self.mapping_dict is not None:
            self.included_classes = list(self.mapping_dict.keys())
            class_ids = list(self.mapping_dict.values())
            sorted_unique_class_ids = sorted(np.unique(class_ids))
            self.class_names = [self.class_names[i] for i in sorted_unique_class_ids]


        self.model = Detector(model = net, device = self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def run(self, batch):
        with torch.no_grad():
            boxes_result = []
            labels_result = []
            scores_result = []
                
          
            preds = self.model.inference_step(batch)

            for idx, outputs in enumerate(preds):
                
                if self.mapping_dict is not None:
                    keep_idx = [enum for enum, i in enumerate(outputs["classes"]) if i in self.included_classes]
                    labels = [self.mapping_dict[int(i)-1] for i in outputs["classes"][keep_idx]]
                    outputs["classes"] = np.array(labels)
                    outputs['scores'] = outputs['scores'][keep_idx]
                    outputs['bboxes'] = outputs['bboxes'][keep_idx]

                boxes = outputs['bboxes'] 

                # Here, labels start from 1, subtract 1 
                labels = outputs['classes']
                scores = outputs['scores']

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        return {
            "boxes": boxes_result, 
            "labels": labels_result,
            "scores": scores_result }