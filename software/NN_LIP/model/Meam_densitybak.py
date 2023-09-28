import numpy as np
import opt_einsum as oe
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_, ones_


#TODO:3.卷积？
class MeaMDensity(nn.Module):

    def __init__(self, rs, inta, params, cutoff, nipsin, ntype, **kwargs):
        super(MeaMDensity, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs           : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta         : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff       : type[float], 截断半径
            nipsin       : type[int], 1, 2, 3, 4 => s, p, d, f
            params       : type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            max_elemtype : type[int], 最大元素类型数
            ntype        : type[int], 参数个数, 对应spdf[10, 4]
        Returns:
            None
        '''
        self.rs1 = nn.Parameter(rs[:,:ntype])
        self.inta1 = nn.Parameter(inta[:,:ntype])
        # self.rs2 = nn.Parameter(rs[:,ntype:])
        # self.inta2 = nn.Parameter(inta[:,ntype:])
        self.params = nn.Parameter(torch.Tensor(params))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.Tensor([nipsin]))
        npara = [1]
        # index_para = [0,0,0,1,1,1....]
        index_para = torch.tensor([], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(3**i)
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * (i - 1)))
        print(index_para)
        self.register_buffer('index_para', index_para)

    def gaussian(self, distances, species_, fcut, rs, inta):
        """计算径向函数部分：exp(-inta*(r-rs)^2)
        Args:
            distances: type[float, 形状(neighbour*numatom*nbatch,1)], 所有近邻原子对的距离列表
            species_ : type[int, 形状(neighbour*numatom*nbatch,)], 所有近邻原子对的i原子类型列表

        Returns:
            tensor   : type[float, 形状(neighbour*numatom*nbatch, nwave)], 所有近邻原子对的径向函数列表
        """
        distances = distances.view(-1, 1)
        radial = torch.zeros((distances.shape[0], rs.shape[1]), dtype=distances.dtype, device=distances.device)
        for itype in range(rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial = torch.exp(-10* inta[itype] * torch.square(distances.index_select(0, ele_index) - rs[itype]))*self.cutoff_cosine(distances.index_select(0, ele_index)).view(-1, 1)
                print(torch.isfinite(inta[itype]))
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
        for ipsin in range(1, int(self.nipsin[0])):
            orbital = torch.einsum("ji,ki -> jki", orbital, dist_vec).reshape(-1, totneighbour)
            angular[num:num + orbital.shape[0]] = orbital
            num += orbital.shape[0]
        return angular

    def radial_density(self, gaussian_part, totnatom, index_neighbour, neighbour_species):
        """计算密度函数部分不同近邻元素
        """
        num_param = self.inta1.shape[1]
        num_ele = self.inta1.shape[0]
        density = torch.zeros((totnatom, num_ele, num_param), dtype=gaussian_part.dtype, device=gaussian_part.device)
        for itype in range(num_ele):
            mask = (neighbour_species == itype)
            part_index = index_neighbour[mask]
            part_orbital = gaussian_part[mask]
            if part_orbital.shape[0] > 0:
                part_density = torch.zeros((totnatom, num_param), dtype=gaussian_part.dtype, device=gaussian_part.device)
                density[:, itype, :] = torch.square(part_density.index_add(0, part_index, part_orbital))
        return density

    def angular_density(self, angular_part, gaussian_part, totnatom, index_neighbour, neighbour_species):
        num_param = self.inta1.shape[1]
        orbital_angular = torch.einsum("ji,ik -> ijk", angular_part, gaussian_part)
        num_ele = self.inta1.shape[0]
        density = torch.zeros((totnatom, num_ele, num_param * int(self.nipsin - 1)), dtype=orbital_angular.dtype, device=orbital_angular.device)
        len_index = self.index_para.shape[0]
        for itype in range(self.rs1.shape[0]):
            mask = (neighbour_species == itype)
            part_index = index_neighbour[mask]
            part_orbital = orbital_angular[mask]
            if part_orbital.shape[0] > 0:
                part_density = torch.zeros((totnatom, len_index, num_param), dtype=orbital_angular.dtype, device=orbital_angular.device)
                part_density.index_add_(0, part_index, part_orbital)
                part_density = torch.square(part_density).permute(1, 0, 2)
                idensity = torch.zeros((int(self.nipsin - 1), totnatom, num_param), dtype=orbital_angular.dtype, device=orbital_angular.device)
                idensity.index_add_(0, self.index_para, part_density)
                density[:, itype, :] = idensity.permute(1, 0, 2).contiguous().flatten(1, 2)
        return density

    def cal_density(self, coordinates, atom_index, shifts, species):
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
        angular_part = self.angular(dist_vec, distances)
        gaussian_part1 = self.gaussian(distances, neighbour_species, self.cutoff_cosine(distances), self.rs1, self.inta1)
        # gaussian_part2 = self.gaussian(distances, neighbour_species, self.cutoff_cosine(distances), self.rs2, self.inta2)
        print(gaussian_part1.max(), gaussian_part1.min())
        density = torch.cat((self.radial_density(gaussian_part1, totnatom, atom_index12[0], neighbour_species), self.angular_density(angular_part, gaussian_part1, totnatom, atom_index12[0], neighbour_species)), dim=2)
        return density

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        """计算原子环境特征

        Args:
            coordinates : type[float, 形状(nbatch, numatom, 3)], 所有原子的坐标列表
            numatoms    : type[int, 形状(nbatch, 1)], 体系中原子的数量
            atom_index  : type[int, 形状(2, nbatch, numatom*neigh)], 所有近邻原子对的原子索引列表 j: 0, i:1
            shifts      : type[float, 形状(nbatch, numatom*neigh, 3)], 所有近邻原子对的位移向量列表
            species     : type[int, 形状(nbatch, numatom)], 所有原子的类型列表, 数值见ele_map 已经压平

        Returns:
            tenser: type[float, 形状(nbatch*numatom, nfeature)], 所有原子的环境特征列表
        """
        density = self.cal_density(coordinates, atom_index, shifts, species)
        density_ = torch.einsum("ijk,j->ijk", density, self.params)
        ele_density = torch.sqrt(torch.einsum("ilj,imj->ijlm", density_, density_)).flatten(2, 3).sum(-1)
        return ele_density

class RadialDensity(nn.Module):

    def __init__(self, rs, inta, cutoff, params=None, **kwargs):
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
        self.inta = nn.parameter.Parameter(torch.Tensor(inta))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.params = nn.parameter.Parameter(torch.Tensor(params))

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
        for itype in range(self.rs.shape[0]):
            mask = (neighbour_species == itype)
            part_index = atom_index12[0][mask]
            part_orbital = orbital[mask]
            if part_orbital.shape[0] > 0:
                density_ = torch.zeros((totnatom, self.rs.shape[1]), dtype=part_density.dtype, device=part_density.device)
                part_density[:, itype, :] = torch.square(density_.index_add(0, part_index, part_orbital))
        return part_density.flatten(1,2)

class AngularDensity(nn.Module):

    def __init__(self, rs, inta, cutoff, nipsin, params=None, **kwargs):
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
        self.params = nn.parameter.Parameter(torch.Tensor(params))
        self.register_buffer('nipsin', torch.Tensor([nipsin]))
        npara = []
        # index_para = [0,0,0,1,1,1....]
        index_para = torch.tensor([], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(3**i)
            index_para = torch.cat((index_para, torch.ones((npara[i-1]), dtype=torch.long) * (i - 1)))
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
        for ipsin in range(1, int(self.nipsin[0])):
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
                density[:, itype, :] = idensity.permute(1, 0, 2).contiguous().flatten(1, 2)
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
        density = self.angular_density(self.angular(dist_vec, distances), self.gaussian(distances, neighbour_species), totnatom, atom_index12[0], neighbour_species)
        return density.flatten(1,2)
    
class MeaMDensity11(nn.Module):
    def __init__(self, rs1, inta1, rs2, inta2, params, cutoff, nipsin, **kwargs):
        super(MeaMDensity11, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs           : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta         : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff       : type[float], 截断半径
            nipsin       : type[int], 1, 2, 3, 4 => s, p, d, f
            params       : type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            max_elemtype : type[int], 最大元素类型数
            ntype        : type[int], 参数个数
        '''
        self.radial_filter = RadialDensity(rs1, inta1, cutoff, params)
        self.angular_filter = AngularDensity(rs2, inta2, cutoff, nipsin, params)

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        radial_density = self.radial_filter(coordinates, numatoms, atom_index, shifts, species)
        angular_density = self.angular_filter(coordinates, numatoms, atom_index, shifts, species)
        density = torch.cat((radial_density, angular_density), dim=-1)
        return density


class MeaMDensity2(nn.Module):

    def __init__(self, rs, inta, params, cutoff, nipsin, ntype, **kwargs):
        super(MeaMDensity2, self).__init__()
        '''初始化: 导入描述符超参数
        Args:
            rs           : type[float, 形状为(natomtype, nwave)], 描述符超参数rs
            inta         : type[float, 形状为(natomtype, nwave)], 描述符超参数inta
            cutoff       : type[float], 截断半径
            nipsin       : type[int], 1, 2, 3, 4 => s, p, d, f
            params       : type[float], 形状为(natomtype, )], Cij权重: 元素相关的权重
            max_elemtype : type[int], 最大元素类型数
            ntype        : type[int], 参数个数, 对应spdf[10, 4]
        Returns:
            None
        '''
        self.rs = nn.Parameter(rs)
        self.inta = nn.Parameter(inta)
        self.params = nn.Parameter(torch.Tensor(params))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.Tensor([nipsin]))
        npara = [1]
        # index_para = [0,0,0,1,1,1....]
        index_para = torch.tensor([0], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(3**i)
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
                part_radial = torch.exp(-self.inta[itype] * torch.square(distances.index_select(0, ele_index) - self.rs[itype]))
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
                density[:, itype, :] = idensity.permute(1, 0, 2).contiguous().flatten(1, 2)
        return density

    def cal_density(self, coordinates, atom_index, shifts, species):
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
        density = self.angular_density(self.angular(dist_vec, distances, self.cutoff_cosine(distances)), self.gaussian(distances, neighbour_species), totnatom, atom_index12[0], neighbour_species)
        return density

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        """计算原子环境特征

        Args:
            coordinates : type[float, 形状(nbatch, numatom, 3)], 所有原子的坐标列表
            numatoms    : type[int, 形状(nbatch, 1)], 体系中原子的数量
            atom_index  : type[int, 形状(2, nbatch, numatom*neigh)], 所有近邻原子对的原子索引列表 j: 0, i:1
            shifts      : type[float, 形状(nbatch, numatom*neigh, 3)], 所有近邻原子对的位移向量列表
            species     : type[int, 形状(nbatch, numatom)], 所有原子的类型列表, 数值见ele_map 已经压平

        Returns:
            tenser: type[float, 形状(nbatch*numatom, nfeature)], 所有原子的环境特征列表
        """
        density = self.cal_density(coordinates, atom_index, shifts, species)
        density_ = torch.einsum("ijk,j->ijk", density, self.params)
        ele_density = torch.sqrt(torch.einsum("ilj,imj->ijlm", density_, density_)).flatten(2, 3).sum(-1)
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
                part_radial = torch.exp(-self.inta[itype] * torch.square((distances.index_select(0, ele_index) - self.rs[itype]) / self.cutoff))
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

class MeaMDensityPart(MeaMDensity):

    def forward(self, coordinates, numatoms, atom_index, shifts, species):
        density = self.cal_density(coordinates, atom_index, shifts, species)
        return density.flatten(1, 2)
    

#####

def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): 高斯函数的偏移值。
        widths: 高斯函数的宽度值
        centered (bool, optional): 如果这是真的,高斯函数为中心在原点,偏移量被用来作为他们的宽度(例如用于角函数)。

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances - offset
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss

class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered
        print(self.offsets, self.width)

    def forward(self, distances):
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)

class CutCosine(nn.Module):
    def __init__(self, cutoff):
        super(CutCosine, self).__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        return 0.5 * (torch.cos(torch.pi * torch.minimum(distances / self.cutoff, torch.tensor(1.0))) + 1)

class Angular(nn.Module):
    """角度描述符
    """
    def __init__(self, nipsin):
        super(Angular, self).__init__()
        # 初始化
        self.nipsin = nipsin
        npara = [1]
        # index_para = [0,0,0,1,1,1....]
        index_para = torch.tensor([], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(3**i)
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * (i - 1)))
        self.register_buffer('index_para', index_para)

    def forward(self, dist_vec, distances):
        totneighbour = dist_vec.shape[0]
        dist_vec = torch.einsum("ij,i -> ij", dist_vec, 1 / distances).permute(1, 0).contiguous()
        orbital = torch.ones((1, totneighbour), dtype=dist_vec.dtype, device=dist_vec.device)
        angular = torch.empty((self.index_para.shape[0], totneighbour), dtype=dist_vec.dtype, device=dist_vec.device)
        num = 0
        for ipsin in range(1, int(self.nipsin[0])):
            orbital = torch.einsum("ji,ki -> jki", orbital, dist_vec).reshape(-1, totneighbour)
            angular[num:num + orbital.shape[0]] = orbital
            num += orbital.shape[0]
        return angular



class MeaMDensity22(nn.Module):

    def __init__(self, cutoff, nipsin, elements, n_radial, n_angular,  **kwargs):
        super(MeaMDensity22, self).__init__()
        '''初始化: 导入描述符超参数
        '''
        self.cutoff = cutoff
        self.nipsin = nipsin
        self.elements = elements
        self.n_radial = n_radial
        self.n_angular = n_angular

        # 初始化描述符
        self.radial_filters = nn.ModuleDict()
        for element in self.elements:
            for itype, n in zip(["radial", "angular"], [n_radial, n_angular]):
                self.radial_filters[f"{element}_{itype}"] = GaussianSmearing(start=0.0, stop=self.cutoff, n_gaussians=n, trainable=True, centered=True)
        self.angular_filters = Angular(self.nipsin)
        self.cutoff_filter = CutCosine(self.cutoff)


    def forward(self, coordinates, numatoms, atom_index, shifts, species):
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
        density = self.angular_density(self.angular(dist_vec, distances, self.cutoff_cosine(distances)), self.gaussian(distances, neighbour_species), totnatom, atom_index12[0], neighbour_species)

