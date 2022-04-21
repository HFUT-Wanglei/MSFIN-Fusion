"""
the package store the loss that user defined
"""

import torch.nn.functional as F
import torch
# from losses.config import Configuration


def structure_weighted_binary_cross_entropy_with_logits(input, target:torch.Tensor):
    target_pad = F.pad(target,[10,10,10,10],mode='circular')
    weit = torch.abs(F.avg_pool2d(target_pad, kernel_size=21, stride=1, padding=0)-target)
    b,c,h,w = weit.shape
    weit = (weit-weit.view(b,c,-1).min(dim=-1,keepdim=True)[0].unsqueeze(-1)) / (1e-6+weit.view(b,c,-1).max(dim=-1,keepdim=True)[0].unsqueeze(-1)-weit.view(b,c,-1).min(dim=-1,keepdim=True)[0].unsqueeze(-1))
    dx = F.conv2d(F.pad(target, [1, 1, 0, 0], mode='reflect'),
                  torch.FloatTensor([-0.5, 0, 0.5]).view(1, 1, 1, 3).to(target.device), stride=1, padding=0)
    dy = F.conv2d(F.pad(target, [0, 0, 1, 1], mode='reflect'),
                  torch.FloatTensor([-0.5, 0, 0.5]).view(1, 1, 3, 1).to(target.device), stride=1, padding=0)
    torch.abs_(dx)
    torch.abs_(dy)
    edge_info = (dx + dy) > 0.4
    weit[edge_info] = 0.0
    S_LOSS_GAMA = 3
    weit = 1 + S_LOSS_GAMA * weit
    wbce = F.binary_cross_entropy(input, target, reduction='none')
    wbce = (weit*wbce)
    return wbce.sum()


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def wbce_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # # pred  = torch.sigmoid(pred)
    # inter = ((pred*mask)*weit).sum(dim=(2,3))
    # union = ((pred+mask)*weit).sum(dim=(2,3))
    # wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()


if __name__ == '__main__':
    import sys
    sys.argv.append('-d')
    sys.argv.append('SOD')
    sys.argv.append('-save')
    sys.argv.append('test')
    a = torch.zeros(1,1,240,240).float()
    b = a+0.0001
    print(structure_weighted_binary_cross_entropy_with_logits(a,b))
    print(wbce_loss(a, b))
