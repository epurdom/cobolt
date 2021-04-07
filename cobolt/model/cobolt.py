
import numpy as np
import os
from scipy import sparse
import random
import itertools
from cobolt.model.coboltmodel import CoboltModel
from cobolt.utils import MultiomicDataset

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from typing import List

class Cobolt:

    def __init__(self,
                 dataset: MultiomicDataset,
                 n_latent: int,
                 device = None,
                 lr: float = 0.001,
                 annealing_epochs: int = 30,
                 alpha: float = None,
                 hidden_dims: List = None,
                 intercept_adj: bool = True,
                 slope_adj: bool = True,
                 train_prop: float = 1):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.lr = lr
        self.epoch = 0
        self.annealing_epochs = annealing_epochs
        self.history = {"loss": []}

        self.dataset = dataset
        self.alpha = 50.0 / n_latent if alpha is None else alpha
        self.hidden_dims = [128, 64] if hidden_dims is None else hidden_dims
        self.model = CoboltModel(
            in_channels=dataset.get_feature_shape(),
            hidden_dims=hidden_dims,
            n_dataset=dataset.n_dataset,
            latent_dim=n_latent,
            intercept_adj=intercept_adj,
            slope_adj=slope_adj,
            log=True
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.test_train_split(train_prop)
        self.train_omic = self.get_train_omic()

    def train(self, num_epochs=100):
        for epoch in range(1, num_epochs + 1):
            print('.', end='')
            if self.epoch < self.annealing_epochs:
                annealing_factor = float(self.epoch) / float(self.annealing_epochs)
            else:
                annealing_factor = 1.0

            this_loss = []
            for omics in self.train_omic:
                dt_loader = DataLoader(
                    dataset=self.dataset,
                    batch_size=128,
                    collate_fn=lambda x: collate_wrapper(x, omics),
                    sampler=SubsetRandomSampler(
                        np.intersect1d(self.dataset.get_comb_idx(omics), self.train_idx)
                    ))
                for x in dt_loader:
                    x, omic_combn = x
                    # Forward pass
                    x = [[x_i.to(self.device) if x_i is not None else None for x_i in y] for y in x]
                    latent_loss, recon_loss = self.model(x, elbo_combn=[omic_combn])
                    # Backprop and optimize
                    loss = annealing_factor * latent_loss + recon_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    this_loss.append(latent_loss.item() + recon_loss.item())

            self.history['loss'].append(this_loss)
            self.epoch += 1

            if np.isnan(self.history['loss'][-1].mean()):
                raise ValueError("DIVERGED.")

    def get_latent(self, omic_combn, data="train"):
        return self._get_latent_helper(omic_combn, data, what="latent")

    def get_topic_prop(self, omic_combn, data="train"):
        return self._get_latent_helper(omic_combn, data, what="topic_prop")

    def _get_latent_helper(self, omic_combn, data="train", what="latent"):
        if data == "train":
            sample_idx = self.train_idx
        elif data == "test":
            sample_idx = self.test_idx
        else:
            raise ValueError

        sample_idx = np.intersect1d(self.dataset.get_comb_idx(omic_combn), sample_idx)
        dl = DataLoader(
            dataset=Subset(self.dataset, sample_idx),
            batch_size=128,
            collate_fn=lambda x: collate_wrapper(x, omic_combn),
            shuffle=False
        )
        latent = []
        for i, x in enumerate(dl):
            x = [[x_i.to(self.device) if x_i is not None else None for x_i in y] for y in x]
            if what == "latent":
                latent += [self.model.get_latent(x, elbo_bool=omic_combn)]
            elif what == "topic_prop":
                latent += [self.model.get_topic_prop(x, elbo_bool=omic_combn)]
        res = np.concatenate(latent)
        return res

    def get_train_omic(self, sample=5):
        n_omic = len(self.dataset.omic)
        if n_omic + 1 + sample < 2 ** n_omic - 1:
            train_omic = [[False] * n_omic for i in range(n_omic)]
            for i in range(n_omic):
                train_omic[i][i] = True
            train_omic.append([True] * n_omic)
            i = 0
            while i < sample:
                s = random.sample(range(n_omic), random.choice(range(1, n_omic)))
                s = [True if i in s else False for i in range(n_omic)]
                if not s in train_omic:
                    train_omic.append(s)
                    i += 1
        else:
            train_omic = [list(i) for i in itertools.product([False, True], repeat=n_omic)]
            train_omic = [i for i in train_omic if any(i)]
        return train_omic

    def test_train_split(self, train_prop):
        n_samples = len(self.dataset)
        permuted_idx = np.random.permutation(n_samples)
        self.train_idx = permuted_idx[:int(n_samples * train_prop)]
        self.test_idx = permuted_idx[int(n_samples * train_prop):]
        self.barcode = self.dataset.get_barcode()
        self.train_barcode = self.barcode[self.train_idx]


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
