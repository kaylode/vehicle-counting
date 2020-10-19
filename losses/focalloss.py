import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
"""
class FocalLoss(nn.Module):
    
    def __init__(self,alpha=None, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob) 
        target = target_tensor.long()
        return {'T':F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor.long(), 
            weight=self.weight,
            reduction = self.reduction
        ).mean()}



if __name__ == '__main__':
    device = torch.device('cuda')
    criterion = FocalLoss().to(device)
    criterion2 = FocalLoss2().to(device)
    preds = torch.rand(5,20).to(device)
    targets = torch.randint(0,20,(5,)).to(device)
    loss = criterion(preds, targets)
    loss2 = criterion2(preds, targets)
    #loss.backward()
    print(loss.item())
    print(loss2.item())


