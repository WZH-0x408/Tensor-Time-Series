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
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load configs
        self.config = configs
        # Task configs
        task_configs = configs['Task']
        mode = task_configs['mode']
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

    def epoch_train(self)->dict:
        info = {
            'train/loss': 0
        }
        self.model.train()
        for (data, gt) in self.train_loader:
            data = data.to(self.device)
            gt = gt.to(self.device)
            self.model.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, gt)
            loss.backward()
            self.optim.step()
            info['train/loss'] = loss.item()
        return info

    def epoch_valid(self)->dict:
        info = {}
        self.model.eval()
        pred_list = []
        gt_list = []
        with torch.no_grad():
            for (data, gt) in self.valid_loader:
                data = data.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred, gt)
                info['valid/loss'] = loss.item()
                gt = gt.cpu().numpy().tolist()
                pred = pred.cpu().numpy().tolist()
                gt_list.extend(gt)
                pred_list.extend(pred)
        # bench mark
        self.benchmark.eval(pred_list, gt_list)
        print(f"valid loss: {info['valid/loss']}")
        return info

    def epoch_early_stop(self, info:dict):
        if info['valid/loss'] < self.best_loss:
            self.best_loss = info['valid/loss']
            self.early_stop_count = 0
            self.save_model()
        else:
            self.early_stop_count += 1
        if self.early_stop_count >= self.early_stop:
            # print(...)
            return True
        return False

    def save_model(self, path):
        pass

    def train(self):
        # prepare?
        info = {}
        # to device
        self.model.to(self.device)
        # start training
        for epoch in range(self.max_epoch):
            info['epoch'] = epoch
            # epoch train and 
            train_info = self.epoch_train()
            # if valid ?
            valid_info = self.epoch_valid()
            # update info
            # self.logger.update(train_info)
            # self.logger.update(valid_info)
        # train - end
    
