import torch
import torch.nn as nn
import os
from datetime import datetime

class Checkpoint():
    """
    Checkpoint for saving model state
    :param save_per_epoch: (int)
    :param path: (string)
    """
    def __init__(self, save_per_iter = 1000, path = None):
        self.path = path
        self.save_per_iter = save_per_iter
        # Create folder
        if self.path is None:
            self.path = os.path.join('weights',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        

    def save(self, model, **kwargs):
        """
        Save model and optimizer weights
        :param model: Pytorch model with state dict
        """
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        epoch = kwargs['epoch'] if 'epoch' in kwargs else '0'
        iters = kwargs['iters'] if 'iters' in kwargs else '0'
        model_path = "_".join([model.model_name,str(epoch), str(iters)])
        if 'interrupted' in kwargs:
            model_path +='_interrupted'
            
        weights = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
        }

        torch.save(weights, os.path.join(self.path,model_path)+".pth")
    
def load_checkpoint(model, path):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """
    state = torch.load(path)
    try:
        model.model.load_state_dict(state["model"])
        model.optimizer.load_state_dict(state["optimizer"])
    except KeyError:
        try:
            ret = model.model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
    print("Loaded Successfully!")
