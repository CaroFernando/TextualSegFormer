import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss
        

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):    
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class NELoss(nn.Module):
    def __init__(self, alpha, beta):
        super(NELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.floss = FocalLoss()
        self.dloss = DiceLoss()

    def forward(self, y_hat, y):
        return self.alpha * self.floss(y_hat, y) + self.beta * self.dloss(y_hat, y)
    
class IoULoss(nn.Module):
    def __init__(self, threshold):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits with values betwen 0 and 1

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # binary values
        inputs = (inputs > self.threshold).float()
        targets = (targets > self.threshold).float()
        
        intersection = (inputs * targets).sum()                            
        total = (inputs*inputs).sum() + (targets*targets).sum() - intersection
        
        return 1 - (intersection + smooth)/(total + smooth)
    