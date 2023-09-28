# from NN_LIP.interface.Meam_density import MeaMDensityPart, RadialDensity, AngularDensity
from NN_LIP.model.Meam_density import MeaMDensityPart, RadialDensity, AngularDensity
from NN_LIP.model.MODEL import ElementNNModel, SingleNNModel
from NN_LIP.model.Property_energy import PESmodel as base_model
import pickle
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ase
from ase.io import read, write
from NN_LIP.data.get_neighbour import neighbor_pairs

ele_map = {"Ta": 1, "He": 0}
atom_en = np.array([0.00168913, -2.24248703])

def get_data(filename):
    atoms = read(filename, format='vasp')
    cutoff = 5.0
    coordinates = atoms.get_positions()
    numatoms = atoms.get_global_number_of_atoms()
    cell = atoms.get_cell()
    species = atoms.get_chemical_symbols()
    pbc = np.array(atoms.get_pbc(), dtype=np.double)
    species = [ele_map[i] for i in species]
    pbc = torch.tensor(pbc).float()
    coordinates = torch.tensor([coordinates]).float()
    species = torch.tensor([species]).float()
    cell = torch.tensor(cell).float()
    numatoms = torch.tensor([numatoms])
    atom_index, shifts, maxneigh = neighbor_pairs(pbc, coordinates, species, cell, cutoff, 27)
    return coordinates, numatoms, atom_index.reshape(1,2,-1), shifts.reshape(1,-1,3), species


nipsin = 1
outputneuron = 1
atomtype = ["He", "Ta"]

actfunc = "Celu"
with open("./describ_params.pkl", "rb") as f:
    radial_rs, radial_inta, angular_rs, angular_inta = pickle.load(f)
cutoff = torch.tensor(5.0)
n_radial = radial_inta.shape[1]
n_angular = angular_inta.shape[1]
elements = 2
nl = [(n_radial+n_angular*nipsin)*2, 64, 32]
radial_filter = RadialDensity(radial_rs, radial_inta, cutoff)
angular_filter = AngularDensity(angular_rs, angular_inta, cutoff, nipsin)
mdescrib = MeaMDensityPart(radial_filter, angular_filter)
nnmod = ElementNNModel(outputneuron, atomtype, nl, actfunc, bias=False)
model = base_model(mdescrib, nnmod)
model.load_weights("./params/ckpt/best_model.pt")


# coordinates, numatoms, atom_index, shifts, species = get_data("./pos/He4.vasp")
# epred = model.predict((coordinates, numatoms, atom_index, shifts, species))
# print(epred)

# atom_index1 = atom_index.permute(1, 0, 2).contiguous()
# tmp_index = torch.arange(coordinates.shape[0], device=coordinates.device) * coordinates.shape[1]
# self_mol_index = tmp_index.view(-1, 1).expand(-1, atom_index1.shape[2]).reshape(1, -1)
# coordinates_ = coordinates.flatten(0, 1)
# totnatom = coordinates_.shape[0]
# padding_mask = torch.nonzero((shifts.view(-1, 3) > -1e9).all(1)).view(-1)
# atom_index12 = (atom_index1.view(2, -1) + self_mol_index).index_select(1, padding_mask)
# selected_cart = coordinates_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
# shift_values = shifts.view(-1, 3).index_select(0, padding_mask)
# dist_vec = selected_cart[0] - selected_cart[1] + shift_values
# neighbour_species = species[0].index_select(0, atom_index12[0])
# print(dist_vec.shape, atom_index.shape, species.shape, neighbour_species.shape)
# model_lmp = torch.jit.load("./TaHe_float.pt")
# epred_lmp = model_lmp(dist_vec, atom_index[0].t(), species[0], neighbour_species)
# print(epred_lmp)




# 读入h5
h5 = h5py.File("./TaHe.hdf5")
data = h5['rawdata']
natoms_list = h5.attrs['natoms_list']
# label, path, species, numtaoms, energy, e_li, energy, err
df = pd.DataFrame( columns=['label', 'path', 'numatoms', 'energy', 'e_li', "deltaE", "deltaE_pred", 'err'])
for natoms in natoms_list:
    print(natoms)
    tmp_data = data[str(natoms)]
    label = tmp_data['datatype']
    path = tmp_data['filepath']
    coordinates, numatoms, atom_index, shifts, species, e_li, energy,deltaE = tmp_data['coordinates'], tmp_data['numatoms'], tmp_data['atom_index'], tmp_data['shifts'], tmp_data['species'], tmp_data['e_li'], tmp_data['energy'], tmp_data['deltaE']
    coordinates = torch.from_numpy(np.array(coordinates)).float()
    numatoms = torch.from_numpy(np.array(numatoms)).int()
    atom_index = torch.from_numpy(np.array(atom_index)).int()
    shifts = torch.from_numpy(np.array(shifts)).float()
    species = torch.from_numpy(np.array(species)).int()
    epred = model.predict((coordinates, numatoms, atom_index, shifts, species))
    df = pd.concat([df, pd.DataFrame({'label': label, 'path': path, 'numatoms': numatoms.detach().numpy(), 'energy': energy, 'e_li': e_li, "deltaE": deltaE, "deltaE_pred": epred.detach().numpy(), 'err': np.abs(deltaE-epred.detach().numpy())})], ignore_index=True)
    df.to_csv("./tmp.csv")
df.to_csv("./TaHe.csv")

# data = pd.read_csv("./tmp.csv", header=0)
# # 去掉Ta_lat, TaHe_lat
# data = data[data['label'] != 'Ta_lat']
# data = data[data['label'] != 'TaHe_lat']
# # x轴为energy, y轴为avg_err, 颜色为label
# energy = data['energy'] / data['numatoms']
# avg_err = np.abs((data['deltaE']- data['deltaE_pred'])/data['numatoms'])
# # 画图
# sns.scatterplot(x=energy, y=avg_err,  hue=data['label'])
# # 横着放legend
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()
