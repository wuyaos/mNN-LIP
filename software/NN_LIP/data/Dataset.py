# 创建数据集
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from NN_LIP.model.Meam_density import MeaMDensity

COMPRESSION = 'gzip'


# todo：分割数据集
def split_data(h5file: str,
               ratio: list = [0.8, 0.1, 0.1],
               shuffle: bool = True,
               seed: int = 24,
               target="full",
               compression=COMPRESSION):
    h5 = h5py.File(h5file, 'r+')
    num_data = h5['dataset']['uuid'].len()
    if shuffle:
        np.random.seed(seed)
        index = np.random.permutation(num_data)
    else:
        index = np.arange(num_data)
    # index
    train_index = index[:int(num_data * ratio[0])]
    val_index = index[int(num_data * ratio[0]):int(num_data * (ratio[0] + ratio[1]))]
    test_index = index[int(num_data * (ratio[0] + ratio[1])):]
    index_dict = {'train': train_index, 'val': val_index, 'test': test_index}
    print(f"分割数据集：train-val-test\n{len(train_index)}-{len(val_index)}-{len(test_index)}")
    # 加载数据
    uuid = h5['dataset']['uuid'][()]
    datatype = h5['dataset']['datatype'][()]
    coordinates = h5['dataset']['coordinates'][()]
    species = h5['dataset']['species'][()]
    numatoms = h5['dataset']['numatoms'][()]
    if target == 'full':
        energy = h5['dataset']['energy'][()].reshape(-1) / numatoms.reshape(-1)
        forces = h5['dataset']['forces'][()]
    elif target == 'delta':
        energy = h5['dataset']['deltaE'][()].reshape(-1) / numatoms.reshape(-1)
        forces = h5['dataset']['deltaF'][()]
    else:
        print("target参数错误")
    atom_index = h5['dataset']['atom_index'][()]
    shifts = h5['dataset']['shifts'][()]

    # 创建group
    try:
        del h5['split']
    except:
        pass

    h5.create_group('split')
    h5['split'].create_group('train')
    h5['split'].create_group('val')
    h5['split'].create_group('test')
    h5['split'].attrs['ratio'] = ratio
    h5['split'].attrs['seed'] = seed
    h5['split'].attrs['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    dt = h5py.string_dtype(encoding='utf-8')
    # 分割数据集
    for tag in index_dict.keys():
        print(f"正在创建{tag}数据集")
        select_index = index_dict[tag]
        uuid_ = uuid[select_index]
        datatype_ = datatype[select_index]
        coordinates_ = coordinates[select_index]
        species_ = species[select_index]
        energy_ = energy[select_index]
        forces_ = forces[select_index]
        atom_index_ = atom_index[select_index]
        shifts_ = shifts[select_index]
        numatoms_ = numatoms[select_index]
        # 创建dataset
        h5['split'][tag].create_dataset('uuid', data=uuid_, compression=compression)
        h5['split'][tag].create_dataset('datatype',
                                        data=datatype_,
                                        dtype=dt,
                                        compression=compression)
        h5['split'][tag].create_dataset('coordinates', data=coordinates_, compression=compression)
        h5['split'][tag].create_dataset('species', data=species_, compression=compression)
        h5['split'][tag].create_dataset('energy', data=energy_, compression=compression)
        h5['split'][tag].create_dataset('forces', data=forces_, compression=compression)
        h5['split'][tag].create_dataset('atom_index', data=atom_index_, compression=compression)
        h5['split'][tag].create_dataset('shifts', data=shifts_, compression=compression)
        h5['split'][tag].create_dataset('numatoms', data=numatoms_, compression=compression)
    h5.close()


class MyDataLoader(Dataset):

    def __init__(self, h5file: str, tag: str = 'train', core: bool = True):
        self.h5file = h5file
        self.tag = tag
        self.core = core
        with h5py.File(h5file, 'r') as f:
            self.length = f['split'][tag]['coordinates'].len(
            )  # to get the length, do not load the data

    def __len__(self):
        return self.length

    def open_hdf5(self):
        # 读到内存中，加快读取速度, 若爆内存则去掉[()]和driver='core'
        self.data_hdf5 = h5py.File(self.h5file, 'r')
        if self.core:
            self.coordinates = self.data_hdf5['split'][self.tag]['coordinates'][()]
            self.species = self.data_hdf5['split'][self.tag]['species'][()]
            self.atom_index = self.data_hdf5['split'][self.tag]['atom_index'][()]
            self.shifts = self.data_hdf5['split'][self.tag]['shifts'][()]
            self.numatoms = self.data_hdf5['split'][self.tag]['numatoms'][()]
            self.energy = self.data_hdf5['split'][self.tag]['energy'][()]
        else:
            self.coordinates = self.data_hdf5['split'][self.tag]['coordinates']
            self.species = self.data_hdf5['split'][self.tag]['species']
            self.atom_index = self.data_hdf5['split'][self.tag]['atom_index']
            self.shifts = self.data_hdf5['split'][self.tag]['shifts']
            self.numatoms = self.data_hdf5['split'][self.tag]['numatoms']
            self.energy = self.data_hdf5['split'][self.tag]['energy']

    def __getitem__(self, item: int):
        if not hasattr(self, 'data_hdf5'):
            self.open_hdf5()
        numatoms = torch.tensor([self.numatoms[item]])
        coordinates = torch.from_numpy(self.coordinates[item]).float()
        species = torch.from_numpy(self.species[item])
        atom_index = torch.from_numpy(self.atom_index[item])
        shifts = torch.from_numpy(self.shifts[item]).float()
        energy = torch.tensor(self.energy[item], dtype=torch.float32)
        return (coordinates, numatoms, atom_index, shifts, species), energy

    def __del__(self):
        if hasattr(self, 'data_hdf5'):
            self.data_hdf5.close()


class MyDataLoader_forces(Dataset):

    def __init__(self, h5file: str, tag: str = 'train', core: bool = True):
        self.h5file = h5file
        self.tag = tag
        self.core = core
        with h5py.File(h5file, 'r') as f:
            self.length = f['split'][tag]['coordinates'].len(
            )  # to get the length, do not load the data

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.data_hdf5 = h5py.File(self.h5file, 'r')
        if self.core:
            # 读到内存中，加快读取速度, 若爆内存则去掉[()]和driver='core', 多线程
            self.coordinates = self.data_hdf5['split'][self.tag]['coordinates'][()]
            self.species = self.data_hdf5['split'][self.tag]['species'][()]
            self.forces = self.data_hdf5['split'][self.tag]['forces'][()]
            self.atom_index = self.data_hdf5['split'][self.tag]['atom_index'][()]
            self.shifts = self.data_hdf5['split'][self.tag]['shifts'][()]
            self.numatoms = self.data_hdf5['split'][self.tag]['numatoms'][()]
            self.energy = self.data_hdf5['split'][self.tag]['energy'][()]
        else:
            self.coordinates = self.data_hdf5['split'][self.tag]['coordinates']
            self.species = self.data_hdf5['split'][self.tag]['species']
            self.forces = self.data_hdf5['split'][self.tag]['forces']
            self.atom_index = self.data_hdf5['split'][self.tag]['atom_index']
            self.shifts = self.data_hdf5['split'][self.tag]['shifts']
            self.numatoms = self.data_hdf5['split'][self.tag]['numatoms']
            self.energy = self.data_hdf5['split'][self.tag]['energy']

    def __getitem__(self, item: int):
        if not hasattr(self, 'data_hdf5'):
            self.open_hdf5()
        numatoms = torch.tensor([self.numatoms[item]])
        coordinates = torch.from_numpy(self.coordinates[item]).float()
        species = torch.from_numpy(self.species[item])
        forces = torch.from_numpy(self.forces[item]).float()
        atom_index = torch.from_numpy(self.atom_index[item])
        shifts = torch.from_numpy(self.shifts[item]).float()
        energy = torch.tensor(self.energy[item], dtype=torch.float32)
        return (coordinates, numatoms, atom_index, shifts, species), (energy, forces)

    def __del__(self):
        if hasattr(self, 'data_hdf5'):
            self.data_hdf5.close()


def load_dataset(h5file: str,
                 batch_size=32,
                 multiprocessing_context=None,
                 num_workers=16,
                 train_force=False,
                 drop_last=False,
                 core: bool = True):
    if train_force:
        train = MyDataLoader_forces(h5file, 'train', core=core)
        test = MyDataLoader_forces(h5file, 'val', core=core)
    else:
        train = MyDataLoader(h5file, 'train', core=core)
        test = MyDataLoader(h5file, 'val', core=core)
    data_train = DataLoader(train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            multiprocessing_context=multiprocessing_context,
                            pin_memory=True,
                            persistent_workers=True,
                            drop_last=drop_last)
    data_test = DataLoader(test,
                           batch_size=batch_size,
                           shuffle=False,
                           multiprocessing_context=multiprocessing_context,
                           num_workers=4,
                           pin_memory=True,
                           persistent_workers=True,
                           drop_last=drop_last)
    return data_train, data_test


#TODO:需要考虑爆内存的问题
def get_dataset_norm(h5file,
                     describ_params,
                     accelerator=None,
                     outfile: str = "dataset_norm.json",
                     multiprocessing_context=None,
                     num_workers=12,
                     batch_size=32):
    rs, inta, cutoff, nipsin, params = describ_params
    mydescrib = MeaMDensity(rs, inta, cutoff, nipsin, params)

    # 加载数据集
    data_train, _ = load_dataset(h5file,
                                 batch_size,
                                 multiprocessing_context=multiprocessing_context,
                                 num_workers=num_workers,
                                 train_flag=False)

    if accelerator is not None:
        mydescrib, data_train = accelerator.prepare(mydescrib, data_train)

    density = np.array([], dtype=np.float32).reshape(0, nipsin * inta.shape[1])
    print("开始计算描述符")
    with trange(len(data_train), leave=True, dynamic_ncols=False, file=sys.stdout,
                smoothing=0) as t:
        for i, ((coordinates, numatoms, atom_index, shifts, species), _) in enumerate(data_train):
            # 计算描述符
            species = species.view(-1)
            density_ = mydescrib(coordinates, numatoms, atom_index, shifts, species)
            mask = (species > -0.5)
            com_index = torch.nonzero(mask).view(-1)
            density_ = density_[com_index]
            density = np.concatenate((density, density_.cpu().numpy()), axis=0)
            t.set_postfix(batch=i)
            t.update
    print("描述符计算完成")
    # 计算均值和方差
    mean = np.mean(density, axis=0)
    std = np.std(density, axis=0)
    # 写入文件
    import json
    with open(outfile, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"均值和方差写入完成: mean: {mean}, std: {std}")