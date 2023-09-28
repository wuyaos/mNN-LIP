import torch
import numpy as np
import os
from NN_LIP.lammps.Meam_density import MeaMDensity, MeaMDensityPart
from NN_LIP.model.MODEL import SingleNNModel, ElementNNModel
import torch.nn as nn


class PESModel(nn.Module):

    def __init__(self, density_filter, nnmod_filter,**kwargs):
        super(PESModel, self).__init__()
        self.density = density_filter
        self.nnmod = nnmod_filter

    def forward(self, coordinates, atom_index, local_species, neigh_species):
        atom_index = atom_index.t().contiguous()
        coordinates.requires_grad_(True)
        density = self.density(coordinates, atom_index, local_species, neigh_species)
        output = self.nnmod(density, local_species)
        varene = torch.sum(output)
        grad = torch.autograd.grad([varene,], [coordinates,])[0]
        if grad is not None:
            return varene.detach(), -grad.detach().view(-1), output.detach()
