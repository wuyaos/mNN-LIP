import torch
import torch.nn as nn

# 自定义损失函数

class MyLoss(nn.Module):
    def __init__(self, loss_fn):
        super(MyLoss, self).__init__()
        self.loss_fn = loss_fn
        self.prop_ceff = torch.ones(2, dtype=torch.float32)

    def update(self, prop_ceff):
        self.prop_ceff = prop_ceff

    def forward(self, output, target):
        mask = (target[1] > -1e9).all(dim=2)
        output_ = [output[0], output[1][mask]]
        target_ = [target[0], target[1][mask]]
        loss_ = torch.cat([iloss_fn(ioutput,itarget).view(-1) for ioutput, itarget, iloss_fn in zip(output_,target_,self.loss_fn)])
        loss = torch.sum(torch.mul(loss_, self.prop_ceff.to(loss_.device)))
        loss_detail = {
            "loss": loss,
            "energy_loss": loss_[0],
            "force_loss": loss_[1],
            "pe": self.prop_ceff[0],
            "pf": self.prop_ceff[1],
            }
        return loss_detail

class ForceLoss(nn.Module):
    def __init__(self, delta):
        super(ForceLoss, self).__init__()
        self.delta = delta

    def forward(self, output, target):
        delta_force = torch.abs(output - target)
        # loss < delta: 0, loss >= delta: loss;mask: 对应loss >= delta
        mask = (delta_force > self.delta).float()
        loss = torch.sum(torch.square(delta_force * mask)) / torch.sum(mask)
        return loss
