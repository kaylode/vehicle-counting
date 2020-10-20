import time
import torch
import os
import cv2
from tqdm import tqdm
import argparse
import numpy as np
import json
from PIL import Image
from torch.backends import cudnn
from matplotlib import colors
import matplotlib.pyplot as plt


from models.backbone import EfficientDetBackbone
from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import *

coco_vehicle_id_map = {
        1: 0,
        2: 1,
        3: 0,
        5: 2,
        7: 3
    }

custom_vehicle_id_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }

frame_width = 1280
frame_height = 720

def main(args):
    path = args.path
    min_conf = args.min_conf
    min_iou = args.min_iou
    coef = args.c
    batch_size = args.batch_size
    weight_path = args.weight
    output_vid = args.output
    start = args.frame_start
    end = args.frame_end
    saved_path = args.saved_path
    box_include = args.box

    compound_coef = coef
    force_input_size = None  # set None to use default size

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = min_conf
    iou_threshold = min_iou

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    outvid = None
    if output_vid is not None:
        outvid = cv2.VideoWriter(output_vid,cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    if weight_path.endswith('efficientdet-d2.pth'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                                ratios=anchor_ratios, scales=anchor_scales)
        vehicle_id = coco_vehicle_id_map
    else:
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=4,
                             ratios=anchor_ratios, scales=anchor_scales)
        vehicle_id = custom_vehicle_id_map

    state = torch.load(weight_path, map_location='cpu')
    if isinstance(state, dict):
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    print('Load pretrained weights!')

    
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    

    # Read images

    img_folder = path[-10:-4]
    video_name = img_folder[-6:]


    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    if not os.path.exists(saved_path+'/{}'.format(video_name)):
        os.mkdir(saved_path+'/{}'.format(video_name))

 
    vidcap = cv2.VideoCapture(path)
    obj_track = {}

    frame_idx = 0
    with tqdm(total=13500) as pbar:
        while(vidcap.isOpened()):
            ims = []
            for b in range(batch_size):
                success, frame = vidcap.read()
                if not success:
                    break
                ims.append(frame)

            torch.cuda.empty_cache()
            ori_imgs, framed_imgs, framed_metas = preprocess(ims, max_size=input_size)
            
            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            
            with torch.no_grad():
                _ ,regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold, vehicle_id = vehicle_id)
                outs = invert_affine(framed_metas, out)

                del regression
                del classification
                del anchors
                
                
                for idx, out in enumerate(outs):
                    bbox_xyxy, cls_conf, cls_ids = out['rois'], out['scores'], out['class_ids']
                    
                    out_dict = {
                        'rois': bbox_xyxy.tolist(),
                        'class_ids': cls_ids.tolist(),
                        'scores': cls_conf.tolist()
                    }

                    if box_include:
                        with open(saved_path+'/{}/{}'.format(video_name,str(frame_idx).zfill(5)+'.txt'), 'w') as f:
                            for i,j in zip(out_dict['class_ids'], out_dict['rois']):
                                f.write('{} {} {} {} {}\n'.format(i, int(j[0]),int(j[1]),int(j[2]),int(j[3])))

                    else:
                        with open(saved_path+'/{}/{}'.format(video_name,str(frame_idx).zfill(5)+'.json'), 'w') as f:
                            json.dump(out_dict, f)
                    frame_idx+=1
            display_img(outs, ims, imshow=False, outvid=outvid, vehicle_id = vehicle_id)
                
            del x
            pbar.update(batch_size)

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--path', type=str, 
                        help='path to video')
    parser.add_argument('--min_conf', type=float, default= 0.3,
                        help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.3,
                        help='minimum iou threshold for non max suppression')
    parser.add_argument('-c', type=int, default = 2,
                        help='version of EfficentDet')

    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',
                        help='version of EfficentDet')

    parser.add_argument('--batch_size', type=int, default = 2,
                        help='batch size')
    
    parser.add_argument('--output', type=str, default = None,
                        help='name of output to .avi file')

    parser.add_argument('--frame_start', type=int, default = 0,
                        help='start from frame')

    parser.add_argument('--frame_end', type=int, default = 13500,
                        help='end at frame')

    parser.add_argument('--saved_path', type=str, default = 'results/detection',
                        help='save detection at')

    parser.add_argument('--box', type=bool, default = False,
                        help='save detection at') 
    args = parser.parse_args()                    
    main(args)
    