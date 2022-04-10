
import numpy as np
import os
from scipy import sparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import random
import itertools
from xgboost import XGBRegressor
import numpy as np
from cobolt.model.coboltmodel import CoboltModel
from cobolt.model.clustering import ClusterUtil
from cobolt.utils import MultiomicDataset
import umap
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class Cobolt:
    """
    A Cobolt model

    Parameters
    ----------
    dataset
        A MultiomicDataset object.
    n_latent
        Number of latent variables used in the Cobolt model.
    device
        The device on which the model will be trained, such as 'cpu' or 'cuda'.
        If not specified, the device will be set to 'cuda' if available.
    lr
        Learning rate for the Adam optimizer.
    annealing_epochs
        Number of annealing epochs for the cost annealing scheme.
    alpha
        Parameter of the Dirichlet prior distribution.
    hidden_dims
        A list of integers indicating the number of hidden dimensions to use for
        the encoder neural networks. The number of fully connected layers are
        determined by the length of the list.
    intercept_adj
        Whether to use the intercept term for batch correction.
    slope_adj
        Whether to use the slope term for batch correction.
    train_prop
        The proportion of random samples to use for training.
    """
    def __init__(self,
                 dataset: MultiomicDataset,
                 n_latent: int,
                 device: str = None,
                 lr: float = 0.005,
                 annealing_epochs: int = 30,
                 batch_size: int = 128,
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
        self.batch_size = batch_size
        self.history = {"loss": []}

        self.dataset = dataset
        self.n_latent = n_latent
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

        self.latent_raw = {}
        self.latent = {}
        self.reduction_raw = {}
        self.reduction = {}
        self.cluster_model = None

    def train(self,
              num_epochs: int = 100):
        """
        Function for training the Cobolt model.

        Parameters
        ----------
        num_epochs
            Number of epochs/iterations.
        """
        for epoch in tqdm(range(1, num_epochs + 1)):
            if self.epoch < self.annealing_epochs:
                annealing_factor = float(self.epoch) / float(self.annealing_epochs)
            else:
                annealing_factor = 1.0

            this_loss = []
            for omics in self.train_omic:
                this_idx = np.intersect1d(self.dataset.get_comb_idx(omics), self.train_idx)
                if len(this_idx) == 0:
                    continue
                dt_loader = DataLoader(
                    dataset=self.dataset,
                    batch_size=self.batch_size,
                    collate_fn=lambda x: collate_wrapper(x, omics),
                    sampler=SubsetRandomSampler(this_idx))
                this_size = len(this_idx)
                for x in dt_loader:
                    # Forward pass
                    x = [[x_i.to(self.device) if x_i is not None else None for x_i in y] for y in x]
                    latent_loss, recon_loss = self.model(x, elbo_combn=[omics])
                    # Backprop and optimize
                    loss = annealing_factor * latent_loss + recon_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    this_loss.append((latent_loss.item() + recon_loss.item())/this_size)

            self.history['loss'].append(sum(this_loss))
            self.epoch += 1

            if np.isnan(self.history['loss'][-1]):
                raise ValueError("DIVERGED. Try a smaller learning rate.")

    def get_latent(self, omic_combn, data="train", return_barcode=False):
        return self._get_latent_helper(
            omic_combn, data, what="latent", return_barcode=return_barcode
        )

    def get_topic_prop(self, omic_combn, data="train", return_barcode=False):
        return self._get_latent_helper(
            omic_combn, data, what="topic_prop", return_barcode=return_barcode
        )

    def _get_latent_helper(self,
                           omic_combn,
                           data="train",
                           what="latent",
                           return_barcode=False):
        if data == "train":
            sample_idx = self.train_idx
        elif data == "test":
            sample_idx = self.test_idx
        else:
            raise ValueError

        if self.epoch == 0:
            raise Exception("Model haven't been trained yet.")

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
        if return_barcode:
            return res, self.dataset.get_barcode()[sample_idx]
        return res

    def calc_all_latent(self,
                        target: List[bool] = None):
        """
        Calculate the latent variable estimation.

        Parameters
        ----------
        target
            A list of boolean indicating which posterior distribution is used
            as benchmark for correction.
        """
        n_modality = len(self.dataset.omic)
        if target is None:
            target = [True] * n_modality
        target_dt, target_barcode = self.get_latent(target, return_barcode=True)
        dt_corrected = [target_dt]
        barcode_corrected = target_barcode
        for i, x in enumerate(self.dataset.omic):
            om_combn = [False] * n_modality
            om_combn[i] = True
            raw_dt, raw_barcode = self.get_latent(om_combn, return_barcode=True)
            bool_train = np.isin(raw_barcode, target_barcode)
            bool_test = ~np.isin(raw_barcode, barcode_corrected)
            if sum(bool_test) != 0:
                raw_dt_train = raw_dt[bool_train, ]
                raw_dt_test = raw_dt[bool_test]
                raw_bc_train = raw_barcode[bool_train]
                raw_bc_test = raw_barcode[bool_test]
                barcode_dict = {x: i for i, x in enumerate(raw_bc_train)}
                reorder = [barcode_dict[i] for i in target_barcode]
                raw_dt_train = raw_dt_train[reorder, ]
                this_predicted = []
                for i in range(self.n_latent):
                    xgb_model = XGBRegressor()
                    xgb_model.fit(X=raw_dt_train, y=target_dt[:, i].copy())
                    this_predicted.append(xgb_model.predict(raw_dt_test))
                dt_corrected.append(np.asarray(this_predicted).T)
                barcode_corrected = np.concatenate((barcode_corrected, raw_bc_test))
        dt_corrected = np.vstack(dt_corrected)
        dt_corrected = (dt_corrected.T - np.mean(dt_corrected, axis=1)).T
        self.latent = {
            "latent": dt_corrected,
            "barcode": barcode_corrected,
            "epoch": self.epoch
        }

    def calc_all_latent_raw(self):
        n_modality = len(self.dataset.omic)
        dt, barcode = self.get_latent([True] * n_modality, return_barcode=True)
        posterior = ["joint"] * len(barcode)
        for i, x in enumerate(self.dataset.omic):
            om_combn = [False] * n_modality
            om_combn[i] = True
            raw_dt, raw_barcode = self.get_latent(om_combn, return_barcode=True)
            dt = np.vstack((dt, raw_dt))
            barcode = np.concatenate((barcode, raw_barcode))
            posterior.extend([x] * len(raw_barcode))
        self.latent_raw = {
            "latent": dt,
            "barcode": barcode,
            "posterior": np.asarray(posterior),
            "epoch": self.epoch
        }

    def get_all_latent(self, correction=True):
        """
        Return the latent variable estimation.

        Parameters
        ----------
        correction
            Whether to return the corrected latent variable estimation.

        Returns
        -------
        latent
            Latent variable estimation.
        barcode
            Corresponding cell barcode of the latent variables.
        posterior
            Which posterior distribution is used for latent variable
            estimation. Only provided if correction is set to `False`.
        """
        if correction:
            if not self.latent or self.latent["epoch"] != self.epoch:
                self.calc_all_latent()
            return self.latent["latent"], self.latent["barcode"]
        else:
            if not self.latent_raw or self.latent["epoch"] != self.epoch:
                self.calc_all_latent_raw()
            return self.latent_raw["latent"], self.latent_raw["barcode"], self.latent_raw["posterior"]

    def run_UMAP(self,
                 correction=True,
                 n_components=2,
                 n_neighbors=30,
                 min_dist=0.1,
                 metric='euclidean'):
        """
        Run UMAP on the latent variable estimation.

        Parameters
        ----------
        correction
            Whether to use corrected latent variables.
        n_components
            The dimension of the space to embed into, which is usually set to
            2 or 3.
        n_neighbors
            The size of the neighborhood for UMAP.
        min_dist:
            The effective minimum distance between embeded points.
        metric:
            The metric to use to compute distances for UMAP.

        Notes
        -----
            `n_components`, `n_neighbors`, `min_dist`, and `metric` are UMAP
            parameters. We direct users to python package `umap` for additional
            details.
        """
        print("Running UMAP {} latent variable correction.".format("with" if correction else "without"))
        dt = self.get_all_latent(correction=correction)
        latent = dt[0]
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        embedding = reducer.fit_transform(latent)
        if correction:
            self.reduction["UMAP" + str(n_components)] = {
                "embedding": embedding,
                "barcode": dt[1],
                "epoch": self.epoch
            }
        else:
            self.reduction_raw["UMAP" + str(n_components)] = {
                "embedding": embedding,
                "barcode": dt[1],
                "posterior": dt[2],
                "epoch": self.epoch
            }

    def run_tSNE(self, correction=True, perplexity=30):
        print("Running tSNE {} latent variable correction.".format("with" if correction else "without"))
        dt = self.get_all_latent(correction=correction)
        latent = dt[0]
        embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(latent)
        if correction:
            self.reduction["tSNE"] = {
                "embedding": embedding,
                "barcode": dt[1],
                "epoch": self.epoch
            }
        else:
            self.reduction_raw["tSNE"] = {
                "embedding": embedding,
                "barcode": dt[1],
                "posterior": dt[2],
                "epoch": self.epoch
            }

    def clustering(self, k=20, algo="leiden", resolution=1, seed=0, overwrite=False):
        """
        Run clustering on the corrected latent variables.

        Parameters
        ----------
        k
            Number of nearest neighbors to use in the KNN graph construction.
        algo
            Clustering algorithm to use. Available options are "leiden" or
            "louvain".
        resolution
            Clustering resolution to use for leiden clustering. Not used if algo
            is set to "louvain".
        seed
            Random seed to use for leiden clustering. Not used if algo is set
            to "louvain".
        overwrite
            Whether to overwrite previous results with the same clustering
            parameters.
        """
        if not self.cluster_model or not self.cluster_model.check_version(k, self.epoch):
            self.cluster_model = ClusterUtil(k=k, key=self.epoch)
            dt = self.get_all_latent(correction=True)
            latent = dt[0]
            self.cluster_model.fit(latent)
        if algo == "leiden":
            self.cluster_model.run_leiden(resolution=resolution, seed=seed, overwrite=overwrite)
        elif algo == "louvain":
            self.cluster_model.run_louvain(overwrite=overwrite)
        else:
            raise ValueError("Clustering algorithm not supported.")

    def get_clusters(self, algo="leiden", resolution=1, return_barcode=False):
        """
        Return the clustering results.

        Parameters
        ----------
        algo
            Clustering algorithm to use. Available options are "leiden" or
            "louvain".
        resolution
            Clustering resolution to use for leiden clustering. Not used if algo
            is set to "louvain".
        return_barcode
            Whether to return the cells barcode.

        Returns
        -------
        clusters
            An integer array indicating the clustering results.
        barcode
            An array of cell barcode. Only provided if `return_barcode` is set
            to `True`.
        """
        if not self.cluster_model:
            print("Clustering has not been run yet. Call `clustering` function first.")
        else:
            if return_barcode:
                latent, barcode = self.get_all_latent(correction=True)
                return self.cluster_model.get_clusters(algo, resolution), barcode
            else:
                return self.cluster_model.get_clusters(algo, resolution)

    def scatter_plot(self,
                     reduc="UMAP",
                     algo="leiden",
                     resolution=1,
                     correction=True,
                     annotation=None,
                     s=1,
                     figsize=(10, 5)):
        if correction:
            use_reduc = self.reduction
        else:
            use_reduc = self.reduction_raw

        if reduc == "UMAP":
            if "UMAP2" not in use_reduc or use_reduc["UMAP2"]["epoch"] != self.epoch:
                self.run_UMAP(correction=correction)
            dt = use_reduc["UMAP2"]["embedding"]
            barcode = use_reduc["UMAP2"]["barcode"]
        elif reduc == "tSNE":
            if "tSNE" not in use_reduc or use_reduc["tSNE"]["epoch"] != self.epoch:
                self.run_tSNE(correction=correction)
            dt = use_reduc["tSNE"]["embedding"]
            barcode = use_reduc["tSNE"]["barcode"]
        else:
            raise ValueError("Reduction must be UMAP or tSNE")

        if annotation is None:
            annotation = self.get_clusters(algo, resolution)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            scatter1 = ax1.scatter(dt[:, 0], dt[:, 1], c=annotation, s=s, cmap=cm.rainbow)
            ax1.legend(*scatter1.legend_elements(), loc="upper left", title="Cluster")
            datasource = np.array([self.dataset.dataset[b] for b in barcode])
            for i in np.unique(datasource):
                mask = datasource == i
                ax2.scatter(dt[mask, 0], dt[mask, 1], label=i, s=s)
            ax2.legend(loc="upper left", title="Dataset")
            fig.show()
        else:
            raise NotImplementedError

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
