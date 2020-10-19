import torch.nn as nn
import torch


class smoothCELoss(nn.Module):
    """
    References: https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
    """
    def __init__(self, alpha = 1e-6, ignore_index = None, reduction = "mean", device = None):
        super(smoothCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.device = device
        self.alpha = alpha
        if device is None:
            self.device = torch.device("cpu")
    def forward(self, outputs, targets):
        # Outputs size: batch_size * num_classes
        # Targets size: batch_size

        batch_size, num_classes = outputs.shape
        y_hot = torch.zeros(outputs.shape).to(self.device).scatter_(1, targets.unsqueeze(1) , 1.0)
        y_smooth = (1 - self.alpha) * y_hot + self.alpha / num_classes
        loss = torch.sum(- y_smooth * torch.nn.functional.log_softmax(outputs, -1), -1).sum()

        if self.reduction == "mean":
            loss /= batch_size
  
        return {'T': loss}
        