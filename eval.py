from utils.getter import *
import argparse
import os
import cv2
import matplotlib.pyplot as plt 

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import init_weights
from utils.utils import *


WIDTH, HEIGHT = (1280, 720)

def main(args, config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    val_transforms = get_augmentation(config, types = 'val')
    retransforms = Compose([
        Denormalize(box_transform=False),
        ToPILImage(),
        Resize(size = (WIDTH, HEIGHT))
    ])
    idx_classes = {idx:i for idx,i in enumerate(config.obj_list)}
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

    ori_img = cv2.imread(args.image) #Image.open(args.image).convert('RGB') #
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = val_transforms(img)['img'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        batch = {'imgs': img}
        outs = model.inference_step(batch, args.min_conf, args.min_iou)
        outs = postprocessing(outs, batch['imgs'].cpu()[0], retransforms)[0]

    img_show = draw_boxes_v2(ori_img, outs, idx_classes)
    cv2.imwrite('test.jpg',img_show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--image', type=str,  help='path to image')
    parser.add_argument('--min_conf', type=float, default= 0.3, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.3, help='minimum iou threshold for non max suppression')
    parser.add_argument('-c', type=int, default = 2, help='version of EfficentDet')
    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',help='version of EfficentDet')
    parser.add_argument('--batch_size', type=int, default = 2,  help='batch size')
    parser.add_argument('--output_path', type=str, default = None, help='name of output to .avi file')
    parser.add_argument('--config', type=str, default = None,help='save detection at')

    args = parser.parse_args() 
    config = Config(os.path.join('configs',args.config+'.yaml'))                   
    main(args, config)
    