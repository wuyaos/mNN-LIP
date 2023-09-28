from torch4keras.callbacks import Evaluator
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys


class ForceEvaluator(Evaluator):

    def load_data(self, data_test):
        self.data_test = data_test

    # 重构评价函数
    def evaluate(self):
        e_rmse, f_rmse, rmse, totnumatoms = 0, 0, 0, 0
        # 进度条显示
        self.model.eval()
        for data in tqdm(self.data_test, desc='Valid', leave=True, dynamic_ncols=False, file=sys.stdout, smoothing=0):
            (coordinates, numatoms, atom_index, shifts, species), (energy, force) = data
            pred_energy, force_pred = self.model(coordinates, numatoms, atom_index, shifts, species)
            mask = (force > -1e9).all(dim=2)
            # 准确率 rmse
            e_rmse += (F.mse_loss(pred_energy, energy, reduction='sum')).cpu().detach()
            f_rmse += (F.mse_loss(force_pred[mask], force[mask], reduction='sum')).cpu().detach()
            totnumatoms += numatoms.sum().cpu().detach()
        prop_ceff = self.model.prop_ceff.cpu().detach()
        e_rmse = torch.sqrt(e_rmse / self.data_test.dataset.__len__())
        f_rmse = torch.sqrt(f_rmse / (totnumatoms * 3))
        rmse = prop_ceff[0] * e_rmse + prop_ceff[1] * f_rmse
        return {'test_pesrmse': rmse, 'test_e_rmse': e_rmse, 'test_f_rmse': f_rmse}

class BaseEvaluator(Evaluator):

    def load_data(self, data_test):
        self.data_test = data_test

    # 重构评价函数
    def evaluate(self):
        rmse = 0
        # 进度条显示
        # # self.model.eval()
        for data in tqdm(self.data_test, desc='Valid', leave=True, dynamic_ncols=False, file=sys.stdout, smoothing=0):
            (coordinates, numatoms, atom_index, shifts, species), energy = data
            pred_energy = self.model.predict((coordinates, numatoms, atom_index, shifts, species))
            pred_energy = self.model.predict((coordinates, numatoms, atom_index, shifts, species))
            # 准确率 rmse
            rmse += torch.square(pred_energy-energy).sum().cpu().detach()
        rmse = torch.sqrt(rmse / self.data_test.dataset.__len__())
        return {'test_rmse': rmse}