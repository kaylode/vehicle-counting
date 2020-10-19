# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

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

class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)

        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(args, config):
    
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
    
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(args.data_path, config.project_name), set='images',
                            types= 'val',
                          transform=transforms.Compose([Normalizer(mean=config.mean, std=config.std),
                                                        Resizer(input_sizes[args.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(config.obj_list), compound_coef=args.compound_coef,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))
   
    # load last weights
    if args.load_weights is not None:
        if args.load_weights.endswith('.pth'):
            weights_path = args.load_weights
        else:
            weights_path = get_last_weights(args.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if args.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    model = ModelWithLoss(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    
    for epoch in range(args.num_epochs):
        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue

        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            if iter < step - last_epoch * num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                imgs = data['img']
                annot = data['annot']

                
                imgs = imgs.cuda()
                annot = annot.cuda()

                optimizer.zero_grad()
                cls_loss, reg_loss = model(imgs, annot, obj_list=config.obj_list)
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()

                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue

                loss.backward()
                optimizer.step()

                epoch_loss.append(float(loss))

                progress_bar.set_description(
                    'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                        step, epoch, args.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                        reg_loss.item(), loss.item()))

                step += 1

                if step % args.save_interval == 0 and step > 0:
                    save_checkpoint(model, f'efficientdet-d{args.compound_coef}_{epoch}_{step}.pth')
                    print('checkpoint...')

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue
        scheduler.step(np.mean(epoch_loss))

        if epoch % args.val_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(val_generator):
                with torch.no_grad():
                    imgs = data['imgs']
                    annot = data['labels']

                    if config.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    cls_loss, reg_loss = model(imgs, annot, obj_list=config.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss_classification_ls.append(cls_loss.item())
                    loss_regression_ls.append(reg_loss.item())

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch, args.num_epochs, cls_loss, reg_loss, loss))


            if loss + args.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch

                save_checkpoint(model, f'efficientdet-d{args.compound_coef}_{epoch}_{step}.pth')

            model.train()

            # Early stopping
            if epoch - best_epoch > args.es_patience > 0:
                print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                break



if __name__ == '__main__':
    args = get_args()
    config = Config(args.config)
    sets = train(args, config)
    
