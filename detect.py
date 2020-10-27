from utils.getter import *
import argparse
import os
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import init_weights
from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import *

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
    retransforms = Compose([
        Denormalize(box_transform=False),
        ToPILImage(),
        Resize(size = (frame_width, frame_height))
    ])
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
    model.eval()

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
            im_shows = []
            for b in range(args.batch_size):
                success, frame_ = vidcap.read()
                if not success:
                    break

                frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                ims.append(val_transforms(frame))
                im_shows.append(frame_)
            
            with torch.no_grad():
                batch = {'imgs': torch.stack([i['img'] for i in ims]).to(device)}
                outs = model.inference_step(batch, args.min_conf, args.min_iou)
                outs = postprocessing(outs, batch['imgs'].cpu()[0], retransforms)
                for idx, out in enumerate(outs):
                    bbox_xyxy, cls_conf, cls_ids = out['bboxes'], out['scores'], out['classes']
                    bbox_xyxy = bbox_xyxy.astype(np.int)
                    out_dict = {
                        'bboxes': bbox_xyxy.tolist(),
                        'classes': cls_ids.tolist(),
                        'scores': cls_conf.tolist()
                    }
                    with open(args.saved_path+'/{}/{}'.format(video_name,str(frame_idx).zfill(5)+'.json'), 'w') as f:
                        json.dump(out_dict, f)
                    frame_idx+=1
                display_img(outs, im_shows, imshow=False, outvid=outvid)
            pbar.update(args.batch_size)

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
    