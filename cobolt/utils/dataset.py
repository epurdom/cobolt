
import numpy as np
from scipy import sparse
import random
import torch
from torch.utils.data import DataLoader
from cobolt.utils import MultiData


class MultiomicDataset(torch.utils.data.Dataset):
    def __init__(self, dt: MultiData):
        self.dt = dt._get_data()
        self.omic = list(self.dt.keys())
        self.barcode = self._get_unique_barcode()
        self.n_dataset = [np.unique(self.dt[om]['dataset']).shape[0] for om in self.omic]
        b_dict = {om: {b: i for i, b in enumerate(self.dt[om]['barcode'])} for om in self.omic}
        self.barcode_in_om = {om: {b: (b_dict[om][b] if b in b_dict[om] else None) for b in self.barcode}
                              for om in self.omic}

    def __len__(self):
        """Number of samples in the data"""
        return self.barcode.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        b = self.barcode[index]
        dat = [self.dt[om]['counts'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        dataset = [self.dt[om]['dataset'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        return dat, dataset

    def _get_unique_barcode(self):
        barcode = np.concatenate([self.dt[om]['barcode'] for om in self.omic])
        return np.unique(barcode)

    def get_barcode(self):
        return self.barcode

    def get_comb_idx(self, omic_combn):
        bl = [self.dt[om]['barcode'] for om, include in zip(self.omic, omic_combn) if include]
        b = bl[0]
        for x in bl[1:]:
            b = np.intersect1d(b, x)
        return np.where(np.isin(self.barcode, b))

    def get_feature_shape(self):
        return [self.dt[om]['feature'].shape[0] for om in self.omic]


def collate_wrapper(batch, omic_combn):
    dataset = [x[1] for x in batch]
    batch = [x[0] for x in batch]
    dataset = [torch.tensor(list(x)) if include else None
               for x, include in zip(zip(*dataset), omic_combn)]
    batch = [torch.from_numpy(sparse.vstack(x).toarray()).float() if include else None
             for x, include in zip(zip(*batch), omic_combn)]
    return batch, dataset


def shuffle_dataloaders(dt_loader, dt_type):
    dt_idx = [[i] * len(x) for i, x in enumerate(dt_loader)]
    dt_idx = sum(dt_idx, [])
    random.Random(4).shuffle(dt_idx)
    dt_iter = [iter(x) for x in dt_loader]
    i = 0
    while i < len(dt_idx):
        yield next(dt_iter[dt_idx[i]]), dt_type[dt_idx[i]]
        i += 1
