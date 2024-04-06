import os
import sys
import csv
import wandb
from torch.utils.tensorboard import SummaryWriter

class BackendBase:
    def __init__(self, root_dir, configs={}) -> None:
        self.root_dir = root_dir    # the root directory for log fiels
        self.track_var = set()
        self.track_var_with_colon = []
        self.log_frame = {}
    def add_logger(self, logger_name, track_list:list):
        pass
    def log(self, logger_name, name, var):
        pass
    def update(self, logger_name):
        pass
    def close(self):
        pass

class LocalBackend(BackendBase):
    def __init__(self, root_dir, configs={}) -> None:
        super().__init__(root_dir, configs=configs)
        # self.name = 'root'
        self.csv_write_header = {}  
        self.csv_file_map = {}
        self.log_root_dir = os.path.join(self.root_dir, 'LocalBackend')
        if not os.path.exists(self.log_root_dir):
            os.mkdir(self.log_root_dir)
        # self.add_csv_wrietr(self.log_root_dir, self.name)

    def add_logger(self, logger_name):
        if logger_name not in self.csv_file_map:
            self.log_frame[logger_name] = {}
            self.add_csv_dict_file(self.log_root_dir, logger_name)
    
    def log(self, logger_name, name, var):
        self.log_frame[logger_name][name] = var
        # var_name = f'{logger_name}/{name}'
        # if var_name in self.track_var:
        #     self.log_frame[logger_name][name] = var
        # else:
        #     print(f"{logger_name}/{name} is not in track list. Please check your track list.")
    
    def update(self, logger_name):
        # writer = self.csv_writer_map[logger_name]
        file = self.csv_file_map[logger_name]
        writer = csv.DictWriter(file, fieldnames=self.log_frame[logger_name].keys())
        if self.csv_write_header[logger_name] == False:
            writer.writeheader()
            self.csv_write_header[logger_name] = True
        writer.writerow(self.log_frame[logger_name])
        file.flush()
        for logger_name in self.log_frame:
            self.log_frame[logger_name] = {}
    
    def close(self):
        for name in self.csv_file_map:
            self.close_csv_dict_writer(name)
        self.csv_write_header = {}
        self.csv_file_map = {}

    def add_csv_dict_file(self, path, name):
        if name not in self.csv_file_map:
            self.csv_write_header[name] = False
            csv_path = os.path.join(path, f"{name}.csv")
            file = open(csv_path, 'w')
            self.csv_file_map[name] = file

    def close_csv_dict_writer(self, name):
        if name in self.csv_file_map:
            file = self.csv_file_map[name]
            file.close()

class wandbBackend(BackendBase):
    def __init__(self, root_dir, configs={}) -> None:
        super().__init__(root_dir, configs=configs)
        task_configs = configs['Task']
        project = task_configs['project']
        run_name = task_configs['run']
        wandb.init(dir=root_dir, project=project, name=run_name,config=configs)

    def add_logger(self, logger_name):
        self.log_frame[logger_name] = {}
    
    def log(self, logger_name, name, var):
        self.log_frame[logger_name][name] = var
    
    def update(self, logger_name):
        wandb_log = {}
        for key in self.log_frame[logger_name]:
            new_key = f"{logger_name}/{key}"
            wandb_log[new_key] = self.log_frame[logger_name][key]
        wandb.log(wandb_log)
        self.log_frame[logger_name] = {}
    
    def close(self):
        wandb.finish()
        