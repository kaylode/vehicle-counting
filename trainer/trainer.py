import os
import torch.nn as nn
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint
import numpy as np
from loggers.loggers import Logger
import time
from utils.utils import draw_pred_gt_boxes
from utils.postprocess import change_box_order, box_fusion, postprocessing
import random
from torch.cuda import amp
from utils.cuda import NativeScaler

from augmentations import Denormalize, MEAN, STD

class Trainer():
    def __init__(self,
                config,
                model, 
                trainloader, 
                valloader,
                **kwargs):

        self.cfg = config
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.multiscale_training = config.multiscale
        self.set_attribute(kwargs)
        
    def fit(self, start_epoch = 0, start_iter = 0, num_epochs = 10 ,print_per_iter = None):
        self.num_epochs = num_epochs
        self.num_iters = (num_epochs+1) * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch = int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
        
        self.epoch = start_epoch

        # For one-cycle lr only
        if self.scheduler is not None and self.step_per_epoch:
            self.scheduler.last_epoch = start_epoch - 1

        self.start_iter = start_iter % len(self.trainloader)

        print(f'===========================START TRAINING=================================')
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                self.training_epoch()

                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()

                if self.scheduler is not None and self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.epoch)
                

            except KeyboardInterrupt:   
                self.checkpoint.save(
                    self.model, 
                    save_mode = 'last', 
                    epoch = self.epoch, 
                    iters = self.iters, 
                    best_value=self.best_value,
                    class_names=self.trainloader.dataset.class_names,
                    config=self.cfg)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            
            start_time = time.time()
        
            with amp.autocast(enabled=self.use_amp):
                loss, loss_dict = self.model.training_step(batch)
                if self.use_accumulate:
                    loss /= self.accumulate_steps

            self.model.scaler(loss, self.optimizer)
            
            if self.use_accumulate:
                if (i+1) % self.accumulate_steps == 0 or i == len(self.trainloader)-1:
                    self.model.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and not self.step_per_epoch:
                        self.scheduler.step((self.num_epochs + i) / len(self.trainloader))
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Training/Learning rate': lr}
                        self.logging(log_dict, step=self.iters)
            else:
                self.model.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    # self.scheduler.step()
                    self.scheduler.step((self.num_epochs + i) / len(self.trainloader))
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.iters)
                

            torch.cuda.synchronize()

            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            self.iters = self.start_iter + len(self.trainloader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:
                
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters,loss_string, running_time))
                
                log_dict = {
                    "Training/Batch Loss" : running_loss['T']/ self.print_per_iter,
                    "Training/IOU Loss" : running_loss['IOU'] / self.print_per_iter,
                    "Training/OBJ Loss" : running_loss['OBJ'] / self.print_per_iter,
                    "Training/CLS Loss" : running_loss['CLS'] / self.print_per_iter,
                }
                self.logging(log_dict, step=self.iters)
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.num_iters == self.num_iters - 1):
                print(f'Save model at [{self.iters}|{self.num_iters}] to last.pth')
                self.checkpoint.save(
                    self.model, 
                    save_mode = 'last', 
                    epoch = self.epoch, 
                    iters = self.iters, 
                    best_value=self.best_value,
                    class_names=self.trainloader.dataset.class_names,
                    config=self.cfg)

            if self.multiscale_training:
                self.set_random_scale()
                

    def inference_batch(self, testloader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in testloader:
                outputs = self.model.inference_step(batch)
                if isinstance(outputs, (list, tuple)):
                    for i in outputs:
                        results.append(i)
                else:
                    results = outputs
                break      
        return results

    def inference_item(self, img):
        self.model.eval()

        with torch.no_grad():
            outputs = self.model.inference_step({"imgs": img.unsqueeze(0)})      
        return outputs


    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        print('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.valloader):
                _, loss_dict = self.model.evaluate_step(batch)
                
                for (key,value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time
        metric_dict = self.model.get_metric_values()
        self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        print()
        print("[{}|{}] || {} || Time: {:10.4f} s".format(self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(metric +': ' + str(score), end = ' | ')
        print()
        print('==========================================================================')

        log_dict = {
            "Validation/Epoch Loss" : epoch_loss['T'] / len(self.valloader),
            "Validation/IOU Loss" : epoch_loss['IOU'] / len(self.valloader),
            "Validation/OBJ Loss" : epoch_loss['OBJ'] / len(self.valloader),
            "Validation/CLS Loss" : epoch_loss['CLS'] / len(self.valloader),
        }
        metric_log_dict = {f"Validation/{k}":v for k,v in metric_dict.items()}
        log_dict.update(metric_log_dict)
        self.logging(log_dict, step=self.epoch)

        # Save model gives best mAP score
        if metric_dict['MAP'] > self.best_value:
            self.best_value = metric_dict['MAP']
            self.checkpoint.save(
                self.model, 
                save_mode = 'best', 
                epoch = self.epoch, 
                iters = self.iters, 
                best_value=self.best_value,
                class_names=self.trainloader.dataset.class_names,
                config=self.cfg)

        if self.visualize_when_val:
            self.visualize_batch()
        
    def visualize_batch(self):
        if not os.path.exists('./samples'):
            os.mkdir('./samples')

        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.valloader))
            targets = batch['targets']

            image_names = batch['img_names']
            imgs = batch['imgs']
            img_sizes = batch['img_sizes'].numpy()

            if self.cfg.tta is not None:
                outputs = self.cfg.tta.make_tta_predictions(self.model, batch)
            else:
                outputs = self.model.inference_step(batch)

            for idx in range(len(outputs)):
                img = imgs[idx]
                image_name = image_names[idx]
                image_outname = os.path.join('samples', f'{self.epoch}_{self.iters}_{idx}.jpg')

                pred = postprocessing(
                        outputs[idx], 
                        current_img_size=img_sizes[idx],
                        min_iou=self.cfg.min_iou_val,
                        min_conf=self.cfg.min_conf_val,
                        output_format='xywh',
                        max_dets=self.cfg.max_post_nms,
                        keep_ratio=False,
                        mode=self.cfg.fusion_mode)

                boxes = pred['bboxes']
                labels = pred['classes']
                scores = pred['scores']

                target = targets[idx]
                target_boxes = target['boxes']
                target_labels = target['labels']
                
                if len(boxes) == 0 or boxes is None:
                    continue
                
                if self.cfg.box_format == 'yxyx':
                    target_boxes = change_box_order(target_boxes, 'yxyx2xyxy')
                target_boxes = change_box_order(target_boxes, order='xyxy2xywh')

                denormalize = Denormalize(
                    mean=MEAN, 
                    std=STD
                )
                img = denormalize(img = img)

                pred_gt_imgs = img.copy()
                pred_gt_boxes = [boxes, target_boxes]
                pred_gt_labels = [labels, target_labels]
                pred_gt_scores = scores
                pred_gt_name = image_name

                fig = draw_pred_gt_boxes(
                    image_outname = image_outname, 
                    img = pred_gt_imgs, 
                    boxes = pred_gt_boxes, 
                    labels = pred_gt_labels, 
                    scores = pred_gt_scores,
                    image_name = pred_gt_name,
                    figsize=(10,10))

                self.logger.write_image(f'{self.logger.datetime}/{idx}', fig, step=self.epoch)

    def logging(self, logs, step):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values, step=step)

    def set_accumulate_step(self):
        self.use_accumulate = False
        if self.cfg.total_accumulate_steps > 0:
            self.use_accumulate = True
            self.accumulate_steps = max(round(self.cfg.total_accumulate_steps / self.cfg.batch_size), 1) 

    def set_amp(self):
        self.use_amp = False
        if self.cfg.mixed_precision:
            self.use_amp = True

    def set_random_scale(self):
        self.trainloader.dataset.set_random_scale(random_=True)
        
    def __str__(self):
        s0 =  "##########   MODEL INFO   ##########"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0,s1,s2,s5,s6])

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.scheduler = None
        self.clip_grad = 10.0
        self.logger = None
        self.step_per_epoch = False
        self.evaluate_per_epoch = 1
        self.visualize_when_val = True
        self.best_value = 0.0
        self.set_accumulate_step()
        self.set_amp()

        for i,j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()