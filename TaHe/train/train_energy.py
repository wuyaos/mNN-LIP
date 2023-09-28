import gc
import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch_optimizer as optim_fn
from accelerate import Accelerator
from accelerate.utils import set_seed
from joblib.externals.loky.backend.context import get_context
from NN_LIP.data.Dataset import load_dataset
from NN_LIP.model.Callbacks import CSVLogger, ModelLogger, Tensorboard
from NN_LIP.model.Evaluator import BaseEvaluator
from NN_LIP.model.Meam_density import MeaMDensity, MeaMDensityPart, RadialDensity, AngularDensity
from NN_LIP.model.MODEL import ElementNNModel, SingleNNModel
from NN_LIP.model.Property_energy import PESmodel as base_model
from NN_LIP.model.Property_force import PESmodel as force_model
from torch4keras.callbacks import Checkpoint, EarlyStopping, ReduceLROnPlateau


multiprocessing_context = get_context("loky") if os.name == "nt" else None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
gc.collect()
torch.cuda.empty_cache()
set_seed(44)
outputneuron = 1
accelerator = Accelerator()
accelerator.print(accelerator.device)
pwd = Path.cwd().absolute()
workdir = pwd / "params11"
h5file = pwd / "TaHe.hdf5"
(workdir / "logs").mkdir(parents=True, exist_ok=True)
(workdir / "ckpt").mkdir(parents=True, exist_ok=True)
ele_map = {"Ta": 1, "He": 0}
atomtype = ["He", "Ta"]
atom_en = np.array([0.00168913, -2.24248703])
neicut = 5.0
batch_size = 32
num_workers = 12
epochs = 1000
starter_learning_rate = 0.005
weight_decay = 1e-7
actfunc = "Tanh"
nipsin = 1
with open("./describ_params.pkl", "rb") as f:
    radial_rs, radial_inta, angular_rs, angular_inta = pickle.load(f)
cij = torch.tensor([0.4, 1])
cutoff = torch.tensor(5.0)
n_radial = radial_inta.shape[1]
n_angular = angular_inta.shape[1]
elements = 2
nl = [(n_radial+n_angular*nipsin)*2, 64, 64, 32]
print(nl)
#===============================拟合====================================
radial_filter = RadialDensity(radial_rs, radial_inta, cutoff)
angular_filter = AngularDensity(angular_rs, angular_inta, cutoff, nipsin)
mdescrib = MeaMDensityPart(radial_filter, angular_filter)
nnmod = ElementNNModel(outputneuron, atomtype, nl, actfunc)
model = base_model(mdescrib, nnmod)
loss_fn = torch.nn.HuberLoss(reduction="sum")
loss_name = loss_fn.__class__.__name__
optimizer = optim.Adam(model.parameters(), lr=starter_learning_rate, weight_decay=weight_decay, amsgrad=True)
optimizer_name = optimizer.__class__.__name__
data_train, data_test = load_dataset(h5file, batch_size, multiprocessing_context=multiprocessing_context, num_workers=num_workers, drop_last=True, core=False)
data_train, data_test, loss_fn, optimizer, model = accelerator.prepare(data_train, data_test, loss_fn, optimizer, model)
model.compile(optimizer=optimizer, loss=loss_fn, tqdmbar=True)
evaluator = BaseEvaluator(monitor='test_rmse', mode='min', model_path=str(workdir / "ckpt" / 'best_model.pt'), optimizer_path=str(workdir / "ckpt" / 'best_optimizer.pt'), steps_params_path=str(workdir / "ckpt" / 'best_step_params.pt'))
evaluator.load_data(data_test)
ckpt = Checkpoint(str(workdir / "ckpt" / 'model_{epoch:03d}_{test_rmse:.5f}.pt'), optimizer_path=str(workdir / "ckpt" / 'optimizer_{epoch:03d}_{test_rmse:.5f}.pt'), steps_params_path=str(workdir / "ckpt" / 'steps_params_{epoch:03d}_{test_rmse:.5f}.pt'), step_interval=500)
ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=2, mode='min', min_lr=1e-7, cooldown=5)
tensorboard = Tensorboard(log_dir=workdir / 'tensorboard', interval=1, log_modelparams=True)
earlystop = EarlyStopping(monitor='loss', patience=100, mode='min', min_delta=1e-7)
csv_log = CSVLogger(log_path=workdir / "logs" / "train_log.csv", interval=100)
add_info = f"""Optimizer setting:
    lr: {starter_learning_rate}
    optimizer: {optimizer_name}
    loss:{loss_name}
"""
logger = ModelLogger(workdir / "logs" / 'log.log', summarymode='torchinfo', info=add_info)
model.fit(data_train, epochs=epochs, callbacks=[evaluator, csv_log, logger, tensorboard, ckpt, ReduceLR], batch_size=batch_size)