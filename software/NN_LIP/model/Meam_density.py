import numpy as np
import opt_einsum as oe
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_, ones_



class RadialDensity(nn.Module):

    def __init__(self, rs, inta, cutoff, **kwargs):
        super(RadialDensity, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs    : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta  : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff: type[float], 截断半径
            nipsin: type[int], 描述符类型 以Gs Gp Gd Gf依次添加
            params: type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            rs    : type[bool], rs是否可训练，从kwargs中解析，默认为False
            inta  : type[bool], inta是否可训练，从kwargs中解析，默认为False
            params: type[bool], params是否可训练，从kwargs中解析，默认为False
        Returns:
            None
        '''
        self.rs = nn.parameter.Parameter(torch.Tensor(rs))
        self.inta = nn.parameter.Parameter((inta))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))

    def gaussian(self, distances, species_):
        """计算径向函数部分：exp(-inta*(r-rs)^2/cutoff^2)
        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            species_ : type[int, 形状(neighbour*numatom*nbatch,)], 所有近邻原子对的i原子类型列表

        Returns:
            tensor   : type[float, 形状(neighbour*numatom*nbatch, nwave)], 所有近邻原子对的径向函数列表
        """
        distances = distances.view(-1, 1)
        radial = torch.zeros((distances.shape[0], self.rs.shape[1]), dtype=distances.dtype, device=distances.device)
        for itype in range(self.rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial = torch.exp(-10*self.inta[itype] * torch.square(distances.index_select(0, ele_index) - self.rs[itype]))
                radial.masked_scatter_(mask.view(-1, 1), part_radial)
        return radial

    def cutoff_cosine(self, distances):
        """计算截断函数部分：0.5*(cos(pi*r/cutoff)+1)

        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表

        Returns:
            tensor:  type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的截断函数列表
        """
        return 0.5 * (torch.cos(torch.pi * torch.minimum(distances / self.cutoff, torch.tensor(1.0))) + 1)


    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        """计算原子环境特征

        Args:
            coordinates : type[float, 形状(nbatch, numatom, 3)], 所有原子的坐标列表
            numatoms    : type[int, 形状(nbatch, 1)], 体系中原子的数量
            atom_index  : type[int, 形状(2, nbatch, numatom*neigh)], 所有近邻原子对的原子索引列表
            shifts      : type[float, 形状(nbatch, numatom*neigh, 3)], 所有近邻原子对的位移向量列表
            species     : type[int, 形状(nbatch, numatom)], 所有原子的类型列表, 数值见ele_map 已经压平

        Returns:
            tenser: type[float, 形状(nbatch*numatom, nfeature)], 所有原子的环境特征列表
        """
        atom_index = atom_index.permute(1, 0, 2).contiguous()
        tmp_index = torch.arange(coordinates.shape[0], device=coordinates.device) * coordinates.shape[1]
        self_mol_index = tmp_index.view(-1, 1).expand(-1, atom_index.shape[2]).reshape(1, -1)
        coordinates_ = coordinates.flatten(0, 1)
        totnatom = coordinates_.shape[0]
        padding_mask = torch.nonzero((shifts.view(-1, 3) > -1e9).all(1)).view(-1)
        atom_index12 = (atom_index.view(2, -1) + self_mol_index).index_select(1, padding_mask)
        selected_cart = coordinates_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values = shifts.view(-1, 3).index_select(0, padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec, dim=-1)
        neighbour_species = species.index_select(0, atom_index12[0])
        orbital = self.gaussian(distances, neighbour_species) * self.cutoff_cosine(distances).view(-1, 1)
        part_density = torch.zeros((totnatom, self.rs.shape[0], orbital.shape[1]), dtype=orbital.dtype, device=coordinates.device)
        #ToDO:以i为中心
        for itype in range(self.rs.shape[0]):
            mask = (neighbour_species == itype)
            part_index = atom_index12[1][mask]
            part_orbital = orbital[mask]
            if part_orbital.shape[0] > 0:
                density_ = torch.zeros((totnatom, self.rs.shape[1]), dtype=part_density.dtype, device=part_density.device)
                part_density[:, itype, :] = torch.square(density_.index_add(0, part_index, part_orbital))
        return part_density

class AngularDensity(nn.Module):

    def __init__(self, rs, inta, cutoff, nipsin, **kwargs):
        super(AngularDensity, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs    : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta  : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff: type[float], 截断半径
            nipsin: type[int], 描述符类型 以Gp Gd Gf依次添加
            params: type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            rs    : type[bool], rs是否可训练，从kwargs中解析，默认为False
            inta  : type[bool], inta是否可训练，从kwargs中解析，默认为False
            params: type[bool], params是否可训练，从kwargs中解析，默认为False
        Returns:
            None
        '''
        self.rs = nn.parameter.Parameter(torch.Tensor(rs))
        self.inta = nn.parameter.Parameter(torch.Tensor(inta))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.Tensor([nipsin]))
        npara = []
        # index_para = [0,0,0,1,1,1....]
        index_para = torch.tensor([], dtype=torch.long)
        for i in range(0, int(nipsin)):
            npara.append(3**(i+1))
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * i))
        self.register_buffer('index_para', index_para)

    def gaussian(self, distances, species_):
        """计算径向函数部分：exp(-inta*(r-rs)^2/cutoff^2)
        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            species_ : type[int, 形状(neighbour*numatom*nbatch,)], 所有近邻原子对的i原子类型列表

        Returns:
            tensor   : type[float, 形状(neighbour*numatom*nbatch, nwave)], 所有近邻原子对的径向函数列表
        """
        distances = distances.view(-1, 1)
        radial = torch.zeros((distances.shape[0], self.rs.shape[1]), dtype=distances.dtype, device=distances.device)
        for itype in range(self.rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial = torch.exp(-10*self.inta[itype] * torch.square(distances.index_select(0, ele_index) - self.rs[itype]))
                radial.masked_scatter_(mask.view(-1, 1), part_radial)
        return radial

    def cutoff_cosine(self, distances):
        """计算截断函数部分：0.5*(cos(pi*r/cutoff)+1)

        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表

        Returns:
            tensor:  type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的截断函数列表
        """
        return 0.5 * (torch.cos(torch.pi * torch.minimum(distances / self.cutoff, torch.tensor(1.0))) + 1)

    def angular(self, dist_vec, distances):
        """计算角度函数部分

        Args:
            dist_vec  : type[float, 形状(neighbour*numatom*nbatch,3)], 所有近邻原子对的距离向量列表
            distances : type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            f_cut     : type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的截断函数列表

        Returns:
            tenser: type[float, 形状(neighbour*numatom*nbatch, npara[0]+npara[1]+...+npara[ipsin])], 所有近邻原子对的角度函数列表
        """
        totneighbour = dist_vec.shape[0]
        dist_vec = torch.einsum("ij,i -> ij", dist_vec, 1 / distances).permute(1, 0).contiguous()
        orbital = torch.ones((1, totneighbour), dtype=dist_vec.dtype, device=dist_vec.device)
        angular = torch.empty((self.index_para.shape[0], totneighbour), dtype=dist_vec.dtype, device=dist_vec.device)
        num = 0
        for ipsin in range(0, int(self.nipsin)):
            orbital = torch.einsum("ji,ki -> jki", orbital, dist_vec).reshape(-1, totneighbour)
            angular[num:num + orbital.shape[0]] = orbital
            num += orbital.shape[0]
        return angular

    def angular_density(self, angular_part, gaussian_part, totnatom, index_neighbour, neighbour_species):
        num_ele, num_param = self.inta.shape
        orbital_angular = torch.einsum("ji,ik -> ijk", angular_part, gaussian_part)
        density = torch.zeros((totnatom, num_ele, num_param * int(self.nipsin)), dtype=orbital_angular.dtype, device=orbital_angular.device)
        len_index = self.index_para.shape[0]
        for itype in range(num_ele):
            mask = (neighbour_species == itype)
            part_index = index_neighbour[mask]
            part_orbital = orbital_angular[mask]
            if part_orbital.shape[0] > 0:
                part_density = torch.zeros((totnatom, len_index, num_param), dtype=orbital_angular.dtype, device=orbital_angular.device)
                part_density.index_add_(0, part_index, part_orbital)
                part_density = torch.square(part_density).permute(1, 0, 2)
                idensity = torch.zeros((int(self.nipsin), totnatom, num_param), dtype=orbital_angular.dtype, device=orbital_angular.device)
                idensity.index_add_(0, self.index_para, part_density)
                density[:, itype, :] = idensity.permute(1, 0, 2).contiguous().flatten(1,2)
        return density


    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        """计算原子环境特征

        Args:
            coordinates : type[float, 形状(nbatch, numatom, 3)], 所有原子的坐标列表
            numatoms    : type[int, 形状(nbatch, 1)], 体系中原子的数量
            atom_index  : type[int, 形状(2, nbatch, numatom*neigh)], 所有近邻原子对的原子索引列表
            shifts      : type[float, 形状(nbatch, numatom*neigh, 3)], 所有近邻原子对的位移向量列表
            species     : type[int, 形状(nbatch, numatom)], 所有原子的类型列表, 数值见ele_map 已经压平

        Returns:
            tenser: type[float, 形状(nbatch*numatom, nfeature)], 所有原子的环境特征列表
        """
        atom_index = atom_index.permute(1, 0, 2).contiguous()
        tmp_index = torch.arange(coordinates.shape[0], device=coordinates.device) * coordinates.shape[1]
        self_mol_index = tmp_index.view(-1, 1).expand(-1, atom_index.shape[2]).reshape(1, -1)
        coordinates_ = coordinates.flatten(0, 1)
        totnatom = coordinates_.shape[0]
        padding_mask = torch.nonzero((shifts.view(-1, 3) > -1e9).all(1)).view(-1)
        atom_index12 = (atom_index.view(2, -1) + self_mol_index).index_select(1, padding_mask)
        selected_cart = coordinates_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values = shifts.view(-1, 3).index_select(0, padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec, dim=-1)
        neighbour_species = species.index_select(0, atom_index12[0])
        density = self.angular_density(self.angular(dist_vec, distances), self.gaussian(distances, neighbour_species)*self.cutoff_cosine(distances).view(-1,1), totnatom, atom_index12[1], neighbour_species)
        return density

class MeaMDensityPart(nn.Module):
    def __init__(self, radial_filter, angular_filter,**kwargs):
        super(MeaMDensityPart, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs           : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta         : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff       : type[float], 截断半径
            nipsin       : type[int], 1, 2, 3 => p, d, f
            params       : type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            max_elemtype : type[int], 最大元素类型数
            ntype        : type[int], 参数个数
        '''
        self.radial_filter = radial_filter
        self.angular_filter = angular_filter

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        radial_density = self.radial_filter(coordinates, numatoms, atom_index, shifts, species)
        angular_density = self.angular_filter(coordinates, numatoms, atom_index, shifts, species)
        density = torch.cat((radial_density, angular_density), dim=-1).flatten(1,2)
        return density


class MeaMDensity(MeaMDensityPart):
    def __init__(self, radial_filter, angular_filter,  cij, **kwargs):
        super().__init__(radial_filter, angular_filter, **kwargs)
        self.cij = nn.parameter.Parameter(torch.Tensor(cij))

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        radial_density = self.radial_filter(coordinates, numatoms, atom_index, shifts, species)
        angular_density = self.angular_filter(coordinates, numatoms, atom_index, shifts, species)
        density = torch.cat((radial_density, angular_density), dim=-1) + torch.tensor(1e-10)
        ele_density = torch.einsum('ijk,j->ijk', torch.sqrt(density), self.cij)
        ele_density = torch.einsum("ilj,imj->ijlm", ele_density, ele_density).flatten(2,3).sum(-1)
        return ele_density


#TODO:1. 添加多组Cij参数?
class MeaMDensity3(nn.Module):

    def __init__(self, rs, inta, cutoff, nipsin, params=None, **kwargs):
        super(MeaMDensity3, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs    : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta  : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff: type[float], 截断半径
            nipsin: type[int], 描述符类型 以Gs Gp Gd Gf依次添加
            params: type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            rs    : type[bool], rs是否可训练，从kwargs中解析，默认为False
            inta  : type[bool], inta是否可训练，从kwargs中解析，默认为False
            params: type[bool], params是否可训练，从kwargs中解析，默认为False
        Returns:
            None
        '''
        rs_flag = kwargs.get('rs_flag', False)
        inta_flag = kwargs.get('inta_flag', False)
        params_flag = kwargs.get('params_flag', False)

        if rs_flag:
            self.rs = nn.parameter.Parameter(torch.Tensor(rs))
        else:
            self.register_buffer('rs', torch.Tensor(rs))
        if inta_flag:
            self.inta = nn.parameter.Parameter(torch.Tensor(inta))
        else:
            self.register_buffer('inta', torch.Tensor(inta))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.Tensor([nipsin]))
        npara = [1]
        # index_para = [0, 1,1,1, 2,2,2,2,...]
        index_para = torch.tensor([0], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(np.power(3, i))
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * i))

        self.register_buffer('index_para', index_para)
        if params_flag:
            if params is None:
                self.params = nn.parameter.Parameter(torch.ones((self.rs.shape[0],), dtype=torch.float))
            else:
                self.params = nn.parameter.Parameter(torch.Tensor(params))
        else:
            self.register_buffer('params', torch.Tensor(params))

    def gaussian(self, distances, species_):
        """计算径向函数部分：exp(-inta*(r-rs)^2/cutoff^2)
        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            species_ : type[int, 形状(neighbour*numatom*nbatch,)], 所有近邻原子对的i原子类型列表

        Returns:
            tensor   : type[float, 形状(neighbour*numatom*nbatch, nwave)], 所有近邻原子对的径向函数列表
        """
        distances = distances.view(-1, 1)
        radial = torch.zeros((distances.shape[0], self.rs.shape[1]), dtype=distances.dtype, device=distances.device)
        for itype in range(self.rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial = torch.exp(-10*self.inta[itype] * torch.square(distances.index_select(0, ele_index) - self.rs[itype]))
                radial.masked_scatter_(mask.view(-1, 1), part_radial)
        return radial

    def cutoff_cosine(self, distances):
        """计算截断函数部分：0.5*(cos(pi*r/cutoff)+1)

        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表

        Returns:
            tensor:  type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的截断函数列表
        """
        return 0.5 * (torch.cos(torch.pi * torch.minimum(distances / self.cutoff, torch.tensor(1.0))) + 1)

    def angular(self, dist_vec, distances, f_cut):
        """计算角度函数部分

        Args:
            dist_vec  : type[float, 形状(neighbour*numatom*nbatch,3)], 所有近邻原子对的距离向量列表
            distances : type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            f_cut     : type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的截断函数列表

        Returns:
            tenser: type[float, 形状(neighbour*numatom*nbatch, npara[0]+npara[1]+...+npara[ipsin])], 所有近邻原子对的角度函数列表
        """
        totneighbour = dist_vec.shape[0]
        dist_vec = torch.einsum("ij,i -> ij", dist_vec, 1 / distances).permute(1, 0).contiguous()
        orbital = f_cut.view(1, -1)
        angular = torch.empty(self.index_para.shape[0], totneighbour, dtype=dist_vec.dtype, device=dist_vec.device)
        angular[0] = f_cut
        num = 1
        for ipsin in range(1, int(self.nipsin[0])):
            orbital = torch.einsum("ji,ki -> jki", orbital, dist_vec).reshape(-1, totneighbour)
            angular[num:num + orbital.shape[0]] = orbital
            num += orbital.shape[0]
        return angular


    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        """计算原子环境特征

        Args:
            coordinates : type[float, 形状(nbatch, numatom, 3)], 所有原子的坐标列表
            numatoms    : type[int, 形状(nbatch, 1)], 体系中原子的数量
            atom_index  : type[int, 形状(2, nbatch, numatom*neigh)], 所有近邻原子对的原子索引列表
            shifts      : type[float, 形状(nbatch, numatom*neigh, 3)], 所有近邻原子对的位移向量列表
            species     : type[int, 形状(nbatch, numatom)], 所有原子的类型列表, 数值见ele_map 已经压平

        Returns:
            tenser: type[float, 形状(nbatch*numatom, nfeature)], 所有原子的环境特征列表
        """
        atom_index = atom_index.permute(1, 0, 2).contiguous()
        tmp_index = torch.arange(coordinates.shape[0], device=coordinates.device) * coordinates.shape[1]
        self_mol_index = tmp_index.view(-1, 1).expand(-1, atom_index.shape[2]).reshape(1, -1)
        coordinates_ = coordinates.flatten(0, 1)
        totnatom = coordinates_.shape[0]
        padding_mask = torch.nonzero((shifts.view(-1, 3) > -1e9).all(1)).view(-1)
        atom_index12 = (atom_index.view(2, -1) + self_mol_index).index_select(1, padding_mask)
        selected_cart = coordinates_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values = shifts.view(-1, 3).index_select(0, padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec, dim=-1)
        species_ = species.index_select(0, atom_index12[1])
        species_j = species.index_select(0, atom_index12[0])
        orbital = oe.contract("ji,ik -> ijk", self.angular(dist_vec, distances, self.cutoff_cosine(distances)), self.gaussian(distances, species_), backend="torch")
        params = torch.cat((self.params, torch.zeros((1, ), dtype=self.params.dtype, device=coordinates.device)), 0)
        orb_coeff = params.index_select(0, species_.masked_fill(species_ < -0.5, self.params.shape[0]))
        orb_coeff_j = params.index_select(0, species_j.masked_fill(species_j < -0.5, self.params.shape[0]))
        Cij = orb_coeff_j * orb_coeff
        orbital = oe.contract("ijk,i -> ijk", orbital, Cij, backend="torch").permute(0, 2, 1).contiguous()
        part_density = torch.zeros((totnatom, orbital.shape[1], orbital.shape[2]), dtype=orbital.dtype, device=coordinates.device)
        part_density.index_add_(0, atom_index12[0], orbital)
        part_density = torch.square(part_density)
        density = torch.zeros((int(self.nipsin), totnatom, self.rs.shape[1]), dtype=part_density.dtype, device=coordinates.device)
        density.index_add_(0, self.index_para, part_density.permute(2,0,1))
        density = density.permute(1, 0, 2).contiguous().flatten(1, 2)
        return density

