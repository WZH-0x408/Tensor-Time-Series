import os
import torch
from torch.utils.data import DataLoader
from .Logger import Logger
from models import ModelInt_Class_Select
from benchmark.benchmark_manager import Benchmark

class Tester:
    def __init__(self, configs:dict, logger: Logger) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load configs
        self.configs = configs
        # Task configs
        task_configs = configs['Task']
        # Tester configs
        tester_configs = configs['Tester']
        batch_size = tester_configs['batch_size']
        model_path = tester_configs['model_path']
        if not os.path.exists(model_path):
            raise FileExistsError(f'{model_path} is not a model path')
        # Model Integration configs
        model_int_configs = configs['ModelInt']
        model_int_name = model_int_configs['name']
        dataset_name = model_int_configs['dataset']
        model_args = model_int_configs['model_args']
        # benchmark
        # get ModelInt
        model_int_class = ModelInt_Class_Select(model_int_name)
        model_int = model_int_class(dataset_name)
        # model
        model_class = model_int.get_model_class()
        self.model = model_class(**model_args)
        self.model.load_state_dict(torch.load(model_path))
        # dataset
        testset = model_int.get_dataset(mode='test')
        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        # bench mark
        bench_mark_list = configs['BenchMark']
        self.benchmark = Benchmark(bench_mark_list)
        # logger
        track_list = configs['Logger']['track_list']
        self.test_logger = logger.add_sub_logger()
        self.test_logger.init('test', track_list)


    def test(self):
        info = {}
        self.model.to(self.device)
        self.model.eval()
        pred_list = []
        gt_list = []
        with torch.no_grad():
            for (data, gt) in self.test_loader:
                data = data.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(data)
                gt = gt.cpu().numpy()
                pred = pred.cpu().numpy()
                pred_list.extend(pred)
                gt_list.extend(gt)
        # bench mark
        self.benchmark.eval(pred_list, gt_list)
        info.update(self.benchmark.get_log())
        self.test_logger.epoch_update(info)
        return info
        # test - end

