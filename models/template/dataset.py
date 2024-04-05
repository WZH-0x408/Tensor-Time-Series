import torch
import random
import numpy as np
from torch.utils.data import Dataset

from datasets.dataset_manager import DatasetManager, DatasetAPIBase

class Template_Dataset(Dataset):
    def __init__(self, mode='train') -> None:
        dataset_api = DatasetManager().get_dataset_api('template_dataset')
        N = 1000
        self.dataset = dataset_api(N)
        self.full_idx = list(range(N))
        random.shuffle(self.full_idx)
        self.idx_list = []
        if mode == 'train':
            train_end = int(0.6 * N)
            self.idx_list = self.full_idx[:train_end]
        elif mode == 'valid':
            valid_begin, valid_end = int(0.6*N), int(0.8*N)
            self.idx_list = self.full_idx[valid_begin:valid_end]
        elif mode == 'test':
            test_end = int(0.8*N)
            self.idx_list = self.full_idx[test_end:]
    def __len__(self):
        return len(self.idx_list)
    def __getitem__(self, index):
        item = self.dataset.get_data_by_idx(self.idx_list[index])
        data = item['data']
        label = item['label']
        label = np.eye(2)[label]
        return torch.tensor(data).float(), torch.tensor(label).float()
