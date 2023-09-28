import os

import torch
import torch.nn as nn
from torch4keras.model import BaseModel


#TODO:1.保存模型为可以为lammps使用的格式；2.超参数网格搜索
class PESmodel(BaseModel):

    def __init__(self,
                 density,                   # 原子密度描述符，nn.Module
                 nnmod,                     # 神经网络模型，nn.Module
                 ):
        super(PESmodel, self).__init__()
        self.density = density
        self.nnmod = nnmod
        self.prop_ceff = torch.ones(2, dtype=torch.float32)

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        coordinates.requires_grad = True
        species = species.view(-1)
        density = self.density(coordinates, numatoms, atom_index, shifts, species)
        output = self.nnmod(density, species).view(numatoms.shape[0], -1)
        energy_pred = torch.sum(output, dim=1)
        grad_outputs = torch.ones(numatoms.shape[0], device=coordinates.device)
        force = -torch.autograd.grad(energy_pred,
                                     coordinates,
                                     grad_outputs=grad_outputs,
                                     only_inputs=True,
                                     allow_unused=True,
                                     retain_graph=True)[0].view(numatoms.shape[0], -1, 3)
        energy_pred_ = energy_pred / numatoms.view(-1)
        return energy_pred_, force