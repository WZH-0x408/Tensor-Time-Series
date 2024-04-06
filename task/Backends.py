import os
import sys
import csv
import wandb
from torch.utils.tensorboard import SummaryWriter

class BackendBase:
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir    # the root directory for log fiels
        self.track_var = set()
        self.track_var_with_colon = []
        self.log_frame = {}
    def add_logger(self, logger_name, track_list:list):
        pass
    def log(self, logger_name, name, var):
        pass
    def update(self):
        pass
    def close(self):
        pass

class LocalBackend(BackendBase):
    def __init__(self, root_dir) -> None:
        super().__init__(root_dir)
        # self.name = 'root'
        self.csv_writer_map = {}  
        self.csv_file_map = {}
        self.log_root_dir = os.path.join(self.root_dir, 'LocalBackend')
        if not os.path.exists(self.log_root_dir):
            os.mkdir(self.log_root_dir)
        # self.add_csv_wrietr(self.log_root_dir, self.name)

    def add_logger(self, logger_name, track_list: list):
        if logger_name not in self.csv_writer_map:
            self.log_frame[logger_name] = {}
            self.add_csv_dict_wrietr(self.log_root_dir, logger_name, track_list)
            for name in track_list:
                var_name = f'{logger_name}/{name}'
                self.track_var.add(var_name)
    
    def log(self, logger_name, name, var):
        var_name = f'{logger_name}/{name}'
        if var_name in self.track_var:
            self.log_frame[logger_name][name] = var
        else:
            print(f"{logger_name}/{name} is not in track list. Please check your track list.")
    
    def update(self, logger_name):
        writer = self.csv_writer_map[logger_name]
        file = self.csv_file_map[logger_name]
        writer.writerow(self.log_frame[logger_name])
        file.flush()
    
    def close(self):
        for name in self.csv_writer_map:
            self.close_csv_dict_writer(name)
        self.csv_writer_map = {}
        self.csv_file_map = {}

    def add_csv_dict_wrietr(self, path, name, track_list=[]):
        if name not in self.csv_writer_map:
            csv_path = os.path.join(path, f"{name}.csv")
            file = open(csv_path, 'w')
            csv_writer = csv.DictWriter(file, fieldnames=track_list)
            csv_writer.writeheader()
            self.csv_writer_map[name] = csv_writer
            self.csv_file_map[name] = file

    def close_csv_dict_writer(self, name):
        if name in self.csv_writer_map:
            file = self.csv_file_map[name]
            file.close()
            self.csv_writer_map[name] = None
