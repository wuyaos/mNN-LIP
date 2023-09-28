import os

import torch
import torch.nn as nn
from torch4keras.model import BaseModel
import numpy as np

class PESmodel(BaseModel):

    def __init__(self,
                 density,                   # 原子密度描述符，nn.Module
                 nnmod,                     # 神经网络模型，nn.Module
                 ):
        super(PESmodel, self).__init__()
        self.density = density
        self.nnmod = nnmod

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        species = species.view(-1)
        density = self.density(coordinates, numatoms, atom_index, shifts, species)
        output = self.nnmod(density, species).view(numatoms.shape[0], -1)
        energy_pred = torch.sum(output, dim=1) / numatoms.view(-1)
        return energy_pred

    def clip_parameters(self):
        for name, param in self.unwrap_model().named_parameters():
            if 'rs' in name:
                param.data.clamp_(0, 6)
            elif "inta" in name:
                param.data.clamp_(1e-5, 30)
            elif "params" in name:
                param.data.clamp_(0.05, 10)
            elif "cij" in name:
                param.data.clamp_(0.05, 100)