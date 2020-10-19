import os
import torch.nn as nn
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint
import numpy as np
from loggers.loggers import Logger
from utils.utils import clip_gradient
import time


class Trainer(nn.Module):
    def __init__(self, 
                model, 
                trainloader, 
                valloader,
                **kwargs):

        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.set_attribute(kwargs)
        
        

    def fit(self, num_epochs = 10 ,print_per_iter = None):
        self.num_epochs = num_epochs
        self.num_iters = (num_epochs+1) * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch = int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
     
        print('===========================START TRAINING=================================')      
        for epoch in range(self.num_epochs+1):
            try:
                self.epoch = epoch
                self.training_epoch()

                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()
                        
                    
                if self.scheduler is not None:
                    self.scheduler.step()
                if (epoch % self.checkpoint.save_per_epoch == 0 or epoch == num_epochs - 1):
                    self.checkpoint.save(self.model, epoch = epoch)

            except KeyboardInterrupt:   
                self.checkpoint.save(self.model, epoch = epoch, interrupted = True)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        for i, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            start_time = time.time()
            loss, loss_dict = self.model.training_step(batch)


            if loss.item() == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            
            # 
            if self.clip_grad is not None:
                clip_gradient(self.optimizer, self.clip_grad)

            self.optimizer.step()
            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            iters = len(self.trainloader)*self.epoch+i+1
            if iters % self.print_per_iter == 0:
                
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, iters, self.num_iters,loss_string, running_time))
                self.logging({"Training Loss/Batch" : running_loss['T']/ self.print_per_iter,})
                running_loss = {}
                running_time = 0

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
                (loss, loss_dict), metrics = self.model.evaluate_step(batch)
                
                for (key,value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time
        metric_dict.update(metrics)
        self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        print()
        print("[{}|{}] || {} || Time: {:10.4f} s".format(self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(metric +': ' + str(score), end = ' | ')
        print('==')
        print('==========================================================================')

        log_dict = {"Validation Loss/Epoch" : epoch_loss['T'] / len(self.valloader),}
        log_dict.update(metric_dict)
        self.logging(log_dict)
        
    


    def logging(self, logs):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values)



    def forward_test(self):
        self.model.eval()
        outputs = self.model.forward_test()
        print("Feed forward success, outputs's shape: ", outputs.shape)

    def __str__(self):
        s0 = "---------MODEL INFO----------------"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s3 = "Loss function: " + str(self.criterion)[:-2]
        s4 = "Optimizer: " + str(self.optimizer)
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0,s1,s2,s3,s4,s5,s6])

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.scheduler = None
        self.clip_grad = None
        self.logger = None
        self.evaluate_per_epoch = 1
        for i,j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()