from utils.getter import *
import argparse
import os
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import init_weights

frame_width = 1280
frame_height = 720

def main(args, config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    

    video_name = args.video_path[-10:-4] # cam_01.mp4
    
    if args.output_path is not None:
        outvid = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))
    else:
        outvid = None

    if not os.path.exists(args.saved_path):
        os.mkdir(args.saved_path)
    if not os.path.exists(args.saved_path +'/{}'.format(video_name)):
        os.mkdir(args.saved_path +'/{}'.format(video_name))

    val_transforms = get_augmentation(config, types = 'val')

    NUM_CLASSES = len(config.obj_list)
    net = EfficientDetBackbone(num_classes=NUM_CLASSES, compound_coef=args.c,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))

    model = Detector(
                    n_classes=NUM_CLASSES,
                    model = net,
                    criterion= FocalLoss(), 
                    optimizer= torch.optim.Adam,
                    optim_params = {'lr': 0.1},     
                    device = device)

    if args.weight is not None:                
        load_checkpoint(model, args.weight)
    else:
        print('[Info] initialize weights')
        init_weights(model.model)

    # Start detecting
    vidcap = cv2.VideoCapture(args.video_path)
    obj_track = {}

    frame_idx = 0
    with tqdm(total=args.frame_end) as pbar:
        while(vidcap.isOpened()):
            while frame_idx < args.frame_start:
                success, frame = vidcap.read()

            ims = []
            for b in range(args.batch_size):
                success, frame = vidcap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                if not success:
                    break
                ims.append(val_transforms(frame))
            
            batch = torch.stack([i['img'] for i in ims])

         
            with torch.no_grad():

               
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                out = postprocess(
                    x,
                    anchors, regression, classification,
                    regressBoxes, clipBoxes,
                    threshold, iou_threshold)
    
                del regression
                del classification
                del anchors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--video_path', type=str,  help='path to video')
    parser.add_argument('--min_conf', type=float, default= 0.3, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.3, help='minimum iou threshold for non max suppression')
    parser.add_argument('-c', type=int, default = 2, help='version of EfficentDet')
    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',help='version of EfficentDet')
    parser.add_argument('--batch_size', type=int, default = 2,  help='batch size')
    parser.add_argument('--output_path', type=str, default = None, help='name of output to .avi file')
    parser.add_argument('--frame_start', type=int, default = 0, help='start from frame')
    parser.add_argument('--frame_end', type=int, default = 5892,  help='end at frame')
    parser.add_argument('--saved_path', type=str, default = 'results/detection',help='save detection at')
    parser.add_argument('--config', type=str, default = None,help='save detection at')

    args = parser.parse_args() 
    config = Config(os.path.join('configs',args.config))                   
    main(args, config)
    