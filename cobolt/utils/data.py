

from scipy import io, sparse
import os
import pandas as pd
import numpy as np


class SingleData(object):
    """
    A single modality of a single dataset

    Parameters
    ----------
    feature_name
        Name of the modality
    dataset_name
        Name of the dataset
    feature
        Array of length F containing feature names
    count
        Matrix of dimension BxF containing data counts
    barcode
        Array of length B containing cell barcode
    """
    def __init__(self,
                 feature_name: str,
                 dataset_name: str,
                 feature: np.ndarray,
                 count: sparse.csr.csr_matrix,
                 barcode: np.ndarray):
        self.feature_name = feature_name
        self.dataset_name = dataset_name
        unique_feature, feature_idx = np.unique(feature, return_index=True)
        if len(feature) != len(unique_feature):
            print("Removing duplicated features.")
            feature = unique_feature
            count = count[:, feature_idx]
        self.feature = feature
        self.barcode = barcode
        self.count = count
        self.is_valid()

    @classmethod
    def from_file(cls,
                  path: str,
                  feature_name: str,
                  dataset_name: str,
                  feature_file: str = "features.tsv",
                  count_file: str = "counts.mtx",
                  barcode_file: str = "barcodes.tsv",
                  feature_header=None,
                  barcode_header=None,
                  feature_column: int = 0,
                  barcode_column: int = 0):
        """
        Read single modality of a single dataset from files. By default,
        the files for feature names, cell barcodes, and the count
        matrix are `features.tsv`, `counts.mtx`, and `barcodes.tsv`.

        Parameters
        ----------
        path
            The path to the directory with data files
        feature_name
            Name of the modality
        dataset_name
            Name of the dataset
        feature_file
            Name of a tab-separated file of the feature names
        count_file
            Name of the file of the count matrix
        barcode_file
            Name of a tab-separated file of the barcode
        feature_header
            Row number to use as column names for the feature file. If `None`,
            no column names will be used.
        barcode_header
            Row number to use as column names for the barcode file. If `None`,
            no column names will be used.
        feature_column
            The name or the index of the column stores the feature name
            information in the feature file
        barcode_column
            The name or the index of the column stores the barcoe name
            information in the barcode file

        Returns
        -------
        A SingleOmic object
        """
        count = io.mmread(os.path.join(path, count_file)).T.tocsr().astype(float)
        feature = pd.read_csv(
            os.path.join(path, feature_file),
            header=feature_header, usecols=[feature_column]
        )[0].values.astype('str')
        barcode = pd.read_csv(
            os.path.join(path, barcode_file),
            header=barcode_header, usecols=[barcode_column]
        )[0].values.astype('str')
        return cls(feature_name, dataset_name, feature, count, barcode)

    def __getitem__(self, items):
        x, y = items
        return SingleData(self.feature_name, self.dataset_name, self.feature[x],
                          self.count[x, y], self.barcode[y])

    def __str__(self):
        return "A SingleData object.\n" + \
               "Dataset name: {}. Feature name: {}.\n".format(
                   self.dataset_name, self.feature_name) + \
               "Number of features: {}. Number of cells {}.".format(
                   str(len(self.feature)), str(len(self.barcode)))

    def filter_features(self, min_count=10, min_cell=5, upper_quantile=1, lower_quantile=0):
        feature_count = np.sum(self.count, axis=0)
        feature_n = np.sum(self.count != 0, axis=0)
        bool_quality = np.array(
            (feature_n > min_cell) & (feature_count > min_count) &
            (feature_count >= np.quantile(feature_count, lower_quantile)) &
            (feature_count <= np.quantile(feature_count, upper_quantile))
        ).flatten()
        self.feature = self.feature[bool_quality]
        self.count = self.count[:, bool_quality]

    def filter_cells(self, min_count=10, min_feature=5, upper_quantile=1, lower_quantile=0):
        feature_count = np.sum(self.count, axis=1)
        feature_n = np.sum(self.count != 0, axis=1)
        bool_quality = np.array(
            (feature_n > min_feature) & (feature_count > min_count) &
            (feature_count >= np.quantile(feature_count, lower_quantile)) &
            (feature_count <= np.quantile(feature_count, upper_quantile))
        ).flatten()
        self.barcode = self.barcode[bool_quality]
        self.count = self.count[bool_quality, :]

    def filter_barcode(self, cells):
        bool_cells = np.isin(self.barcode, cells)
        self.count = self.count[bool_cells, :]
        self.barcode = self.barcode[bool_cells]

    def subset_features(self, feature):
        bool_features = np.isin(self.feature, feature)
        self.count = self.count[:, bool_features]
        self.feature = self.feature[bool_features]

    def rename_features(self, feature):
        unique_feature, feature_idx = np.unique(feature, return_index=True)
        if len(feature) != len(unique_feature):
            print("Removing duplicated features.")
            feature = unique_feature
            self.count = self.count[:, feature_idx]
        self.feature = np.array(feature)

    def get_data(self):
        return {self.feature_name: self.count}, {self.feature_name: self.feature}, self.barcode

    def get_dataset_name(self):
        return self.dataset_name

    def is_valid(self):
        if self.count.shape[0] != self.barcode.shape[0]:
            raise ValueError("The dimensions of the count matrix and the barcode array are not consistent.")
        if self.count.shape[1] != self.feature.shape[0]:
            raise ValueError("The dimensions of the count matrix and the barcode array are not consistent.")


class MultiData(object):

    def __init__(self, *single_data):
        self.data = {}
        for dt in single_data:
            ct, ft, bc = dt.get_data()
            for mod in ct.keys():
                if mod not in self.data.keys():
                    self.data[mod] = {
                        'feature': [ft[mod]],
                        'barcode': [bc],
                        'counts': [ct[mod]],
                        'dataset': [dt.get_dataset_name()]
                    }
                else:
                    self.data[mod]['feature'].append(ft[mod])
                    self.data[mod]['barcode'].append(bc)
                    self.data[mod]['counts'].append(ct[mod])
                    self.data[mod]['dataset'].append(dt.get_dataset_name())
        for mod in self.data.keys():
            self.data[mod] = merge_modality(self.data[mod])

    def get_data(self):
        return self.data


def merge_modality(dt):
    batch = [np.zeros(x.shape) + i for i, x in enumerate(dt['barcode'])]
    batch = np.concatenate(batch)
    barcode = np.concatenate(dt['barcode'])

    feature = dt['feature'][0]
    for f in dt['feature'][1:]:
        feature = np.intersect1d(feature, f)

    counts = []
    for i in range(len(dt['counts'])):
        common = np.intersect1d(feature, dt['feature'][i], return_indices=True)
        counts += [dt['counts'][i][:, common[2]]]
    counts = sparse.vstack(counts)

    return {
        'feature': feature,
        'counts': counts,
        'barcode': barcode,
        'dataset': batch,
        'dataset_name': dt['dataset']
    }
