
import numpy as np
from scipy import sparse
import random
import itertools
import torch
from torch.utils.data import DataLoader
from cobolt.utils.data import MultiData


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

    def __getitem__(self, index: int):
        """Generates one sample of data"""
        b = self.barcode[index]
        dat = [self.dt[om]['counts'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        dataset = [self.dt[om]['dataset'][self.barcode_in_om[om][b]] if self.barcode_in_om[om][b] is not None else None
               for om in self.omic]
        return dat, dataset

    def __str__(self):
        n_modality = len(self.omic)
        s1 = "A MultiomicDataset object with {} omics:\n".format(n_modality)
        s2 = "".join(["- {}: {} features, {} cells, {} batches.\n".format(
            om, len(self.dt[om]['feature']), len(self.dt[om]['barcode']), self.n_dataset[i])
            for i, om in enumerate(self.omic)])
        s3 = "Joint cells:\n"
        joint_omic = [list(i) for i in itertools.product([False, True], repeat=n_modality) if sum(i) > 1]
        s4 = "".join(["- {}: {} cells.".format(
            ", ".join([om for i, om in enumerate(self.omic) if om_combn[i]]),
            len(self.get_comb_idx(om_combn)))
            for om_combn in joint_omic])
        return s1 + s2 + s3 + s4

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
        return np.where(np.isin(self.barcode, b))[0]

    def get_feature_shape(self):
        return [self.dt[om]['feature'].shape[0] for om in self.omic]
