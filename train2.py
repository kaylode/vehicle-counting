# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117
from utils.getter import *
import argparse
import datetime
import os
import traceback


import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from configs import Config

from models.backbone import EfficientDetBackbone
from datasets.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from models.efficientdet.loss import FocalLoss
from utils.eff_utils import  get_last_weights, init_weights, boolean_string

from models.detector import Detector

def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='weights/')


    args = parser.parse_args()
    return args


def train(args, config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    training_params = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': args.num_workers}

    val_params = {'batch_size': args.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': args.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(args.data_path, config.project_name), set='images',
                                types= 'train',
                               transform=transforms.Compose([Normalizer(mean=config.mean, std=config.std),
                                                             Augmenter(),        
                                                             Resizer(input_sizes[args.compound_coef])]))
    
    val_set = CocoDataset(root_dir=os.path.join(args.data_path, config.project_name), set='images',
                            types= 'val',
                          transform=transforms.Compose([Normalizer(mean=config.mean, std=config.std),
                                                        Resizer(input_sizes[args.compound_coef])]))
    
    trainloader = DataLoader(training_set, **training_params)
    valloader = DataLoader(val_set, **val_params)

    net = EfficientDetBackbone(num_classes=len(config.obj_list), compound_coef=args.compound_coef,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))
    
    # load last weights
    if args.load_weights is not None:
        if args.load_weights.endswith('.pth'):
            weights_path = args.load_weights
        try:
            ret = net.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(net)

    # freeze backbone if train head_only
    if args.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        net.apply(freeze_backbone)
        print('[Info] freezed backbone')

    model = Detector(
                    n_classes=len(config.obj_list),
                    model = net,
                    criterion= FocalLoss(), 
                    optimizer= torch.optim.SGD,
                    optim_params = {'lr': args.lr, 'momentum': 0.9, 'nesterov': True},     
                    device = device)
    
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=100, path = 'weights/effdet'),
                     logger = Logger(log_dir='loggers/runs/effdet'),
                     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, patience=3, verbose=True),
                     evaluate_per_epoch = 2)

    print(trainer)
    trainer.fit(num_epochs=50, print_per_iter=10)

    

if __name__ == '__main__':
    args = get_args()
    config = Config(args.config)
    sets = train(args, config)
    


"""
if __name__ == "__main__":

    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda")
    print("Using", device)

    # Dataloader
    BATCH_SIZE = 1
    my_collate = trainset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    criterion = FocalLoss
    optimizer = torch.optim.Adam
    #metrics = [AccuracyMetric(decimals=3)]
    
    model = EfficientDetector(
                    n_classes = NUM_CLASSES,
                    optim_params = {'lr': 1e-3},
                    criterion= criterion, 
                    optimizer= optimizer,
                    freeze=True,
                    pretrained='weights/pretrained/efficientdet-d0-fixed.pth',
                 
                    device = device)
    
    #load_checkpoint(model, "weights/ssd-voc/SSD300-10.pth")
    #model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
#                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=5, path = 'weights/effdet'),
                     logger = Logger(log_dir='loggers/runs/effdet'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = 2)
    
    print(trainer)
    
    
    
    trainer.fit(num_epochs=50, print_per_iter=10)
    

    # Inference
"""