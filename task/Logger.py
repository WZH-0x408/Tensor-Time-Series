import pathlib
from .Backends import BackendBase, LocalBackend, wandbBackend

class Logger:
    logger_backend = None
    def __init__(self) -> None:
        self.sub_logger = []        
        # self.set_backend(root_dir, backend)

    def init(self, logger_name:str, track_list=[]):        
        self.logger_name = logger_name
        self.track_list, self.track_with_colon = self.parse_track_list(logger_name, track_list)
        print(self.track_list)
        print(self.track_with_colon)
        Logger.logger_backend.add_logger(logger_name)
    
    @staticmethod
    def close():
        Logger.logger_backend.close()

    @staticmethod
    def set_backend(root_dir , configs={}, backend='local'):
        if Logger.logger_backend != None:
            return 
        if backend == 'local':
            Logger.logger_backend = LocalBackend(root_dir, configs)
        elif backend == 'wandb':
            Logger.logger_backend = wandbBackend(root_dir, configs)
        elif backend == 'tensorboard':
            pass

    def parse_track_list(self, logger_name, track_list):
        logger_track_list = []
        colon_track_list = []
        for var_name in track_list:
            if '/' in var_name:
                var_name_splited = var_name.split('/')
                if len(var_name_splited) == 2:
                    if var_name_splited[0] == logger_name:
                        logger_track_list.append(var_name_splited[1])
                        if ':' in var_name_splited[1]:
                            colon_sp = var_name_splited[1].split(':')
                            colon_track_list.append(colon_sp)
        return logger_track_list, colon_track_list

    def add_sub_logger(self):
        new_logger = Logger()
        self.sub_logger.append(new_logger)
        return new_logger

    def log(self, name, var):
        Logger.logger_backend.log(self.logger_name, name, var)

    def epoch_update(self, local_vars:dict):
        # print(local_vars)
        # print(self.track_list)
        for var_name in local_vars:
            if ':' in var_name:
                # colon case
                var_name_sp = var_name.split(':')
                track_sp = self.search_track(var_name_sp)
                if self.match_track(track_sp, var_name_sp):
                    self.log(var_name, local_vars[var_name])
            elif var_name in self.track_list:
                self.log(var_name, local_vars[var_name])
        Logger.logger_backend.update(self.logger_name)

    def search_track(self, name_sp):
        for track_sp in self.track_with_colon:
            if name_sp[0] == track_sp[0]:
                return track_sp
                
    def match_track(self, track_sp, name_sp):
        if not isinstance(track_sp, list):
            return False
        for i in range(len(track_sp)):
            if track_sp[i] == "*":
                return True
            elif track_sp[i] != name_sp[i]:
                return False
        return True