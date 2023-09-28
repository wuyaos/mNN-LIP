import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch4keras.callbacks import Callback, Checkpoint, Logger

import NN_LIP.interface.PES as pes_model
import NN_LIP.lammps.PES as lmp_pes
import NN_LIP.lammps.script_PES as script_PES


class CSVLogger(Callback):
    '''默认logging
    对于valid/dev和test的日志需要在evaluate之后对log进行赋值，如log['dev_f1']=f1，并在Evaluator之后调用
    若每隔一定steps对验证集评估，则Logger的interval设置成和Evaluater一致或者约数，保证日志能记录到

    :param log_path: str, log文件的保存路径
    :param interval: int, 保存log的间隔
    :param mode: str, log保存的模式, 默认为'a'表示追加
    :param separator: str, 指标间分隔符
    :param verbosity: int, 可选[0,1,2]，指定log的level
    :param name: str, 默认为None
    '''

    def __init__(self, log_path: Path, interval=10, mode='a', separator=',', **kwargs):
        super(CSVLogger, self).__init__(**kwargs)

        self.interval = interval
        self.sep = separator
        save_dir = log_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        self.fh = open(log_path, mode, encoding='utf-8', buffering=1)

    def on_epoch_end(self, global_step, epoch, logs=None):
        log_str = f'{self.sep}'.join([f'{v:.5f}' for k, v in logs.items() if k not in {'size'}])
        self.fh.write(f'epoch,{epoch+1}{self.sep}{log_str}')

    def on_batch_end(self, global_step, local_step, logs=None):
        if global_step == 0:
            self.fh.write(f'method{self.sep}{self.sep.join([k for k in logs.keys() if k not in {"size"}])}\n')
        if (global_step + 1) % self.interval == 0:
            log_str = f'{self.sep}'.join([f'{v:.5f}' for k, v in logs.items() if k not in {'size'}])
            self.fh.write(f'step,{global_step+1}{self.sep}{log_str}\n')

    def on_train_end(self, logs=None):
        self.fh.close()
        self.fh = None


class PropertCeffUpdate(Callback):
    """更新loss权重

    Args:
        Callback (_type_): _description_
    """

    def __init__(
            self,
            start_pref_e=0.8,  # 能量的初始权重
            limit_pref_e=1,  # 能量的最大权重
            start_pref_f=0.01,  # 力的初始权重
            limit_pref_f=1e-5,  # 力的最大权重
    ):
        super(PropertCeffUpdate, self).__init__()
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.starter_learning_rate = None
        self.lr = None
        self.criterion = None
        self.prop_ceff = torch.Tensor([self.start_pref_e, self.start_pref_f])

    def on_train_begin(self, logs=None):
        self.starter_learning_rate = self.model.optimizer.param_groups[0]['lr']
        self.criterion = self.model.criterion
        self.criterion.update(self.prop_ceff)
        self.model.prop_ceff = self.prop_ceff

    def on_batch_begin(self, global_step, local_step, logs=None):
        self.lr = self.model.optimizer.param_groups[0]['lr']
        self.prop_ceff[0] = self.start_pref_e + (self.limit_pref_e - self.start_pref_e) * (self.lr - self.starter_learning_rate) / (1e-8 - self.starter_learning_rate)
        self.prop_ceff[1] = self.start_pref_f + (self.limit_pref_f - self.start_pref_f) * (self.lr - self.starter_learning_rate) / (1e-8 - self.starter_learning_rate)
        self.criterion.update(self.prop_ceff)
        self.model.prop_ceff = self.prop_ceff


class ModelLogger(Logger):
    '''默认logging
    对于valid/dev和test的日志需要在evaluate之后对log进行赋值，如log['dev_f1']=f1，并在Evaluator之后调用
    若每隔一定steps对验证集评估，则Logger的interval设置成和Evaluater一致或者约数，保证日志能记录到

    :param log_path: str, log文件的保存路径
    :param interval: int, 保存log的间隔
    :param mode: str, log保存的模式, 默认为'a'表示追加
    :param separator: str, 指标间分隔符
    :param verbosity: int, 可选[0,1,2]，指定log的level
    :param name: str, 默认为None
    :param summarymode: str, 默认为pl，可选["pl", "torchinfo"]，指定模型summary的方式
    :param base_model: nn.Module, 默认为None，当summarymode为torchinfo时，需要指定base_model
    :param info: str, 默认为None，输出信息
    '''

    def __init__(self, log_path, interval=10, mode='a', separator='\t', verbosity=1, name=None, **kwargs):
        super().__init__(str(log_path), interval, mode, separator, verbosity, name, **kwargs)
        # 进度条参数
        self.summarymode = kwargs.get('summarymode', "pl")
        if self.summarymode not in ["pl", "torchinfo"]:
            raise ValueError(f"summarymode must be in ['pl', 'torchinfo'], but got {self.summarymode}")
        self.base_model = kwargs.get('base_model', None)
        self.info = kwargs.get('info', None)
        self.lr = None

    def on_train_begin(self, logs=None):
        import logging
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level_dict[self.verbosity])
        save_dir = os.path.dirname(self.log_path)

        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(self.log_path, self.mode)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if self.summarymode == "pl":
            from NN_LIP.utils.summary import ModelSummary
            modelsummary_str = str(ModelSummary(self.model, max_depth=-1))
        elif self.summarymode == "torchinfo":
            from torchinfo import summary as torchinfo_summary
            model = self.model if self.base_model is None else self.base_model
            modelsummary_str = str(torchinfo_summary(model, input_data=next(iter(self.trainer.train_dataloader))[0], depth=5))
        self.logger.info('Model Summary'.center(40, '='))
        print("Model Summary:")
        for i in modelsummary_str.split('\n'):
            self.logger.info(i)
        self.logger.info('Add infomation'.center(40, '='))
        if self.info is not None:
            for i in self.info.split('\n'):
                self.logger.info(i)
                print(i)
        self.logger.info('Start Training'.center(40, '='))


class Tensorboard(Callback):
    '''默认Tensorboard
    对于valid/dev和test的Tensorboard需要在evaluate之后对log进行赋值，如log['dev/f1']=f1，并在Evaluator之后调用
    赋值需要分栏目的用'/'进行分隔
    若每隔一定steps对验证集评估，则Tensorboard的interval设置成和Evaluater一致或者约数，保证Tensorboard能记录到

    :param log_dir: Path, tensorboard文件的保存路径
    :param method: str, 控制是按照epoch还是step来计算，默认为'epoch', 可选{'step', 'epoch'}
    :param interval: int, 保存tensorboard的间隔
    :param prefix: str, tensorboard分栏的前缀，默认为'train'
    :param on_epoch_end_scalar_epoch: bool, epoch结束后是横轴是按照epoch还是global_step来记录
    '''

    def __init__(self, log_dir: Path, method='epoch', interval=10, prefix='train', on_epoch_end_scalar_epoch=True, log_modelparams=False, **kwargs):
        super(Tensorboard, self).__init__(**kwargs)
        assert method in {'step', 'epoch'}, 'Args `method` only support `step` or `epoch`'
        self.method = method
        self.interval = interval
        self.prefix = prefix + '/' if len(prefix.strip()) > 0 else ''  # 控制默认的前缀，用于区分栏目
        self.on_epoch_end_scalar_epoch = on_epoch_end_scalar_epoch  # 控制on_epoch_end记录的是epoch还是global_step
        self.log_modelparams = log_modelparams  # 是否记录模型参数

        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))  # prepare summary writer

    def get_modelparams(self, step):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
            self.writer.add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), step)

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self.method == 'epoch':
            # 默认记录的是epoch
            log_step = epoch + 1 if self.on_epoch_end_scalar_epoch else global_step + 1
            self.process(log_step, logs)
            if self.log_modelparams:
                self.get_modelparams(log_step)

    def on_batch_end(self, global_step, local_step, logs=None):
        # 默认记录的是global_step
        if (self.method == 'step') and ((global_step + 1) % self.interval == 0):
            self.process(global_step + 1, logs)
            if self.log_modelparams:
                self.get_modelparams(global_step + 1)

    def process(self, iteration, logs):
        logs = logs or {}
        for k, v in logs.items():
            if k in {'size'}:
                continue
            index = k if '/' in k else f"{self.prefix}{k}"
            self.writer.add_scalar(index, v, iteration)


#TODO:待添加功能
class LmpPesModel(Callback):
    """主要面向PES模型，保存模型为LAMMPS可调用的格式
    """

    def __init__(self, model_path: Path, model_hparams: dict, monitor='perf', mode='max', verbose=1, method='epoch', step_interval=100, **kwargs):
        self.monitor = monitor
        assert mode in {'max', 'min'}, 'Compare performance only support `max/min` mode'
        self.mode = mode
        self.verbose = verbose
        self.best_perf = np.inf if mode == 'min' else -np.inf
        self.method = method
        self.model_path = model_path  # 保存路径
        self.step_interval = step_interval  # method='step'时候生效
        self.model_hparams = model_hparams

    def process(self, logs=None):
        perf = logs
        state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "prop_ceff" in k:
                continue
            name = k
            new_state_dict[name] = v
        nipsin = self.model_hparams['nipsin']
        cutoff = self.model_hparams['cutoff']
        rs = self.model_hparams['rs']
        inta = self.model_hparams['inta']
        cij = self.model_hparams['cij']
        mean = self.model_hparams['mean']
        std = self.model_hparams['std']
        nl = self.model_hparams['nl']
        activate = self.model_hparams['activate']
        atomtype = self.model_hparams['atomtype']
        init_pes = lmp_pes.PES(nipsin, cutoff, rs, inta, cij, mean, std, nl, activate, atomtype)
        filepath = self.model_path.parent
        filename = self.model_path.name
        double_model_path = str(filepath / f"double_{filename}.pt")
        float_model_path = str(filepath / f"float_{filename}.pt")

        # 满足条件
        if ((self.mode == 'max') and (perf[self.monitor] >= self.best_perf)) or ((self.mode == 'min') and (perf[self.monitor] <= self.best_perf)):
            self.best_perf = perf[self.monitor]
            # 保存ckpt
            init_pes.load_state_dict(new_state_dict)
            scripted_pes = torch.jit.script(init_pes)
            for params in scripted_pes.parameters():
                params.requires_grad = False
            scripted_pes.to(torch.float64)
            scripted_pes.save(double_model_path)
            scripted_pes.to(torch.float32)
            scripted_pes.save(float_model_path)

        if self.verbose > 0:
            print(f"\n保存模型{double_model_path}\n")

    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        if self.method == 'epoch':
            self.process(logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        if (self.method == 'step') and ((global_step + 1) % self.step_interval == 0):
            self.process(logs)


#TODO:待添加功能
class FreezModel(Callback):
    """主要面向PES模型，保存模型为LAMMPS可调用的格式
    """

    def __init__(self, model_path: Path, model_hparams: dict, monitor='perf', mode='max', verbose=1, method='epoch', step_interval=100, **kwargs):
        self.monitor = monitor
        assert mode in {'max', 'min'}, 'Compare performance only support `max/min` mode'
        self.mode = mode
        self.verbose = verbose
        self.best_perf = np.inf if mode == 'min' else -np.inf
        self.method = method
        self.model_path = model_path  # 保存路径
        self.step_interval = step_interval  # method='step'时候生效
        self.model_hparams = model_hparams

    def process(self, logs=None):
        perf = logs
        state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "prop_ceff" in k:
                continue
            name = k
            new_state_dict[name] = v
        nlinked = self.model_hparams['nlinked']
        neicut = self.model_hparams['neicut']
        nipsin = self.model_hparams['nipsin']
        cutoff = self.model_hparams['cutoff']
        rs = self.model_hparams['rs']
        inta = self.model_hparams['inta']
        cij = self.model_hparams['cij']
        mean = self.model_hparams['mean']
        std = self.model_hparams['std']
        nl = self.model_hparams['nl']
        activate = self.model_hparams['activate']
        atomtype = self.model_hparams['atomtype']
        init_pes = pes_model.PES(nlinked, neicut, nipsin, cutoff, rs, inta, cij, mean, std, nl, activate, atomtype)
        filepath = self.model_path.parent
        filename = self.model_path.name
        double_model_path = str(filepath / f"double_{filename}.pt")
        float_model_path = str(filepath / f"float_{filename}.pt")

        # 满足条件
        if ((self.mode == 'max') and (perf[self.monitor] >= self.best_perf)) or ((self.mode == 'min') and (perf[self.monitor] <= self.best_perf)):
            self.best_perf = perf[self.monitor]
            # 保存ckpt
            init_pes.load_state_dict(new_state_dict)
            # 固话模型
            scripted_pes = torch.jit.script(init_pes)
            for params in scripted_pes.parameters():
                params.requires_grad = False
            scripted_pes.to(torch.float64)
            scripted_pes.save(double_model_path)
            scripted_pes.to(torch.float32)
            scripted_pes.save(float_model_path)

        if self.verbose > 0:
            print(f"\n保存模型{double_model_path}\n")

    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        if self.method == 'epoch':
            self.process(logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        if (self.method == 'step') and ((global_step + 1) % self.step_interval == 0):
            self.process(logs)