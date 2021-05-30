import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss


class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()
        self.l1_loss = L1Loss(reduce=True)

    def forward(self, recover_img, gt):
        losses = {}
        loss_l1 = self.l1_loss(recover_img[0], gt)
        losses["total_loss"] = loss_l1

        return losses
