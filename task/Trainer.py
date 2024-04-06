import random
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from .Logger import Logger
from models import ModelInt_Class_Select
from benchmark.benchmark_manager import Benchmark

class Trainer:
    def __init__(self, configs:dict, logger:Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load configs
        self.config = configs
        # Task configs
        task_configs = configs['Task']
        self.model_save_path = os.path.join(task_configs['out_dir'], 'model')
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        # Trainer configs
        trainer_config = configs['Trainer']
        batch_size = trainer_config['batch_size']
        self.max_epoch = trainer_config['max_epoch']
        self.early_stop = trainer_config['early_stop']
        self.early_stop_count = 0
        self.best_loss = np.inf
        # Model Integration configs
        model_int_configs = configs['ModelInt']
        model_int_name = model_int_configs['name']
        dataset_name = model_int_configs['dataset']
        model_args = model_int_configs['model_args']
        optim_args = model_int_configs['optim_args']
        # BenchMark list
        bench_mark_list = configs['BenchMark']
        self.benchmark = Benchmark(bench_mark_list)
        # convert str to float
        for key in optim_args:
            optim_args[key] = float(optim_args[key])
        # get ModelInt
        model_int_class = ModelInt_Class_Select(model_int_name)
        model_int = model_int_class(dataset_name)
        # dataset
        trainset = model_int.get_dataset(mode='train')
        validset = model_int.get_dataset(mode='valid')
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
        # model, optim, criterion
        model_class = model_int.get_model_class()
        optim_class = model_int.get_optim_class()
        criterion_class = model_int.get_criterion_class()
        # instance
        self.model = model_class(**model_args)
        optim_args['params'] = self.model.parameters()
        self.optim = optim_class(**optim_args)
        self.criterion = criterion_class()
        # logger
        track_list = configs['Logger']['track_list']
        self.train_logger = logger.add_sub_logger()
        self.train_logger.init('train', track_list)
        self.valid_logger = logger.add_sub_logger()
        self.valid_logger.init('valid', track_list)

    def epoch_train(self, epoch)->dict:
        info = {
            'epoch': epoch,
            'loss': 0
        }
        loss_list = []
        self.model.train()
        for (data, gt) in self.train_loader:
            data = data.to(self.device)
            gt = gt.to(self.device)
            self.model.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, gt)
            loss.backward()
            self.optim.step()
            loss_list.append(loss.item())
        info['loss'] = sum(loss_list)/len(loss_list)
        self.train_logger.epoch_update(info)
        # return info

    def epoch_valid(self, epoch)->dict:
        info = {
            'epoch': epoch
        }
        self.model.eval()
        pred_list = []
        gt_list = []
        loss_list = []
        with torch.no_grad():
            for (data, gt) in self.valid_loader:
                data = data.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred, gt)
                loss_list.append(loss.item())
                gt = gt.cpu().numpy().tolist()
                pred = pred.cpu().numpy().tolist()
                gt_list.extend(gt)
                pred_list.extend(pred)
        # bench mark
        self.benchmark.eval(pred_list, gt_list)
        info.update(self.benchmark.get_log())
        info['loss'] = sum(loss_list) / len(loss_list)
        print(f">> epoch: {epoch}: {info['loss']}")
        stop_flag = self.epoch_early_stop(info['loss'])
        # some vars
        info['early_stop_count'] = self.early_stop
        info['best_loss'] = self.best_loss
        self.valid_logger.epoch_update(info)
        return stop_flag

    def epoch_early_stop(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.early_stop_count = 0
            self.save_model('best_model.pth')
        else:
            self.early_stop_count += 1
        if self.early_stop_count >= self.early_stop:
            return True
        return False

    def save_model(self, model_name):
        if '.pth' not in model_name:
            model_name = f"{model_name}.pth"
        save_path = os.path.join(self.model_save_path, model_name)
        torch.save(self.model.state_dict(), save_path)

    def train(self):
        # prepare?
        info = {}
        # to device
        self.model.to(self.device)
        # start training
        for epoch in range(self.max_epoch):
            info['epoch'] = epoch
            # epoch train and 
            self.epoch_train(epoch)
            # if valid ?
            stop_flag = self.epoch_valid(epoch)
            if stop_flag:
                print(f'The model has not improved. STOP TRAINING.')
                break
        # train - end
    
