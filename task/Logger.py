import pathlib
from .Backends import BackendBase, LocalBackend

class Logger:
    logger_backend = None
    def __init__(self) -> None:
        self.sub_logger = []        
        # self.set_backend(root_dir, backend)

    def init(self, logger_name:str, track_list=[]):        
        self.logger_name = logger_name
        self.track_list = self.parse_track_list(logger_name, track_list)
        Logger.logger_backend.add_logger(logger_name, self.track_list)

    def set_backend(self, root_dir ,backend='local'):
        if Logger.logger_backend != None:
            return 
        if backend == 'local':
            Logger.logger_backend = LocalBackend(root_dir)
        elif backend == 'wandb':
            pass
        elif backend == 'tensorboard':
            pass

    def parse_track_list(self, logger_name, track_list):
        logger_track_list = []
        for var_name in track_list:
            if '/' in var_name:
                var_name_splited = var_name.split('/')
                if len(var_name_splited) == 2:
                    if var_name_splited[0] == logger_name:
                        logger_track_list.append(var_name_splited[1])
        return logger_track_list

    def add_sub_logger(self):
        new_logger = Logger()
        self.sub_logger.append(new_logger)
        return new_logger

    def log(self, name, var):
        Logger.logger_backend.log(self.logger_name, name, var)

    def epoch_update(self, local_vars:dict):
        print(local_vars)
        print(self.track_list)
        for var_name in local_vars:
            if var_name in self.track_list:
                self.log(var_name, local_vars[var_name])
        Logger.logger_backend.update(self.logger_name)

