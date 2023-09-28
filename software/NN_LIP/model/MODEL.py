from collections import OrderedDict
import torch
from torch import nn
from torch.nn import (LayerNorm, Linear, Sequential, BatchNorm1d, SELU, SiLU, Tanh, ReLU, CELU, ELU, GELU)
from torch.nn.init import constant_, xavier_uniform_, zeros_


# 归一化层
class NormLayer(nn.Module):

    def __init__(self, mean, std):
        super(NormLayer, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).view(1, -1))
        self.register_buffer('std', torch.Tensor(std).view(1, -1))

    def forward(self, x):
        return (x - self.mean) / self.std

# 限制输出范围
class LimitLayer(nn.Module):
    def __init__(self, min, max):
        super(LimitLayer, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        # 范围
        a = (self.max - self.min) / 2
        out = torch.tanh(x) * a + (self.max + self.min) / 2
        return out


# 根据函数名获取激活函数
def get_activation(actfunc):
    actfunc = actfunc.lower()
    if actfunc == 'tanh':
        return Tanh()
    elif actfunc == 'silu':
        return SiLU()
    elif actfunc == 'gelu':
        return GELU()
    elif actfunc == 'selu':
        return SELU()
    elif actfunc == "relu":
        return ReLU()
    elif actfunc == "celu":
        return CELU()
    elif actfunc == "elu":
        return ELU()
    else:
        raise ValueError("Unknown activation function: {}".format(actfunc))


class SingleNNModel(torch.nn.Module):

    def __init__(
            self,
            outputneuron,  # 神经网络的输出神经元的数量
            atomtype,  # 所有系统的元素
            nl,  # 神经网络结构
            actfunc="Silu",  # 激活函数
            batchnorm=None,  # 是否使用批归一化, None, 'mean-std', 'batchnorm'
            norm=False,  # 是否使用层归一化
            **kwargs):
        super(SingleNNModel, self).__init__()
        # create the structure of the nn
        self.outputneuron = outputneuron
        modules = []
        with torch.no_grad():
            nl.append(outputneuron)
            nhid = len(nl) - 2
            if batchnorm == 'mean-std':
                modules.append(NormLayer(kwargs.get('mean', None), kwargs.get('std', None)))
            elif batchnorm == 'batchnorm':
                modules.append(BatchNorm1d(nl[0], momentum=0.5))
            for i in range(nhid):
                linear = Linear(nl[i], nl[i + 1])
                xavier_uniform_(linear.weight)
                zeros_(linear.bias)
                modules.append(linear)
                if norm:
                    modules.append(LayerNorm(nl[i + 1]))
                modules.append(get_activation(actfunc))
            linear = Linear(nl[nhid], nl[nhid + 1])
            xavier_uniform_(linear.weight)
            zeros_(linear.bias)
            modules.append(linear)
        self.elemental_nets = Sequential(*modules)

    def forward(self, density, species):
        output = torch.zeros((density.shape[0], self.outputneuron),
                             dtype=density.dtype,
                             device=density.device)
        com_index = torch.nonzero(species > -0.5).reshape(-1)
        output[com_index] = self.elemental_nets(density[com_index])
        return output


class ElementNNModel(torch.nn.Module):

    def __init__(
            self,
            outputneuron,  # 神经网络的输出神经元的数量
            atomtype,  # 所有系统的元素
            nl,  # 神经网络结构
            actfunc="Silu",  # 激活函数
            batchnorm=None,  # 是否使用批归一化, None, 'mean-std', 'batchnorm'
            norm=False,  # 是否使用层归一化
            bias=True,
            **kwargs):

        super(ElementNNModel, self).__init__()
        self.outputneuron = outputneuron
        elemental_nets = OrderedDict()
        with torch.no_grad():
            nl.append(outputneuron)
            nhid = len(nl) - 2
            for ele in atomtype:
                modules = []
                if batchnorm == 'mean-std':
                    modules.append(NormLayer(kwargs.get('mean', None), kwargs.get('std', None)))
                elif batchnorm == 'batchnorm':
                    modules.append(BatchNorm1d(nl[0], momentum=0.7))
                for i in range(nhid):
                    linear = Linear(nl[i], nl[i + 1], bias=bias)
                    xavier_uniform_(linear.weight)
                    if bias:
                        zeros_(linear.bias)
                    modules.append(linear)
                    if norm:
                        modules.append(LayerNorm(nl[i + 1]))
                    modules.append(get_activation(actfunc))
                linear = Linear(nl[nhid], nl[nhid + 1], bias=bias)
                xavier_uniform_(linear.weight)
                if bias:
                    zeros_(linear.bias)
                modules.append(linear)
                # if ele == 'He':
                #     modules.append(LimitLayer(-4, 4))
                # if ele == 'Ta':
                #     modules.append(LimitLayer(-6, 1))
                elemental_nets[ele] = Sequential(*modules)
        self.elemental_nets = nn.ModuleDict(elemental_nets)

    def forward(self, density, species):
        output = torch.zeros((density.shape[0], self.outputneuron), dtype=density.dtype, device=density.device)
        for itype, (_, m) in enumerate(self.elemental_nets.items()):
            ele_index = torch.nonzero(species == itype).view(-1)
            if ele_index.shape[0] > 0:
                output[ele_index] = m(density[ele_index])
        return output