import pathlib
import wandb
import tensorboard
import csv

class Logger:
    def __init__(self, log_name:str, 
                 tracking_list=[] ,logger='none'):
        self.log_name = log_name
        self.tracking_list = tracking_list
        self.logger = logger
        self.infobase = { v_name:[] for v_name in self.tracking_list }
        self.infobase['update_cnt'] = 0
    
    def update(self, info: dict):

        self.infobase['update_cnt'] += 1
        for v_name in self.tracking_list:
            self.infobase[v_name].append(info[v_name])

        if self.logger == 'wandb':
            self.wandb_log(info)
    '''
    wandb logger
    '''
    def wandb_init(self):
        pass    
    def wandb_log(self, info:dict):
        pass    
    def wandb_finished(self):
        pass
    
    '''
    CSV data
    '''
    def saved_as_csv(self):
        pass