

from scipy import io, sparse
import os
import pandas as pd
import numpy as np


class SingleData(object):

    def __init__(self, feature_name, feature, count, barcode):
        """
        Create single-omic data from existing objects.
        """
        self.feature_name = feature_name
        self.feature = feature
        self.barcode = barcode
        self.count = count

    @classmethod
    def from_file(cls, path, feature_name,
                  feature_file="features.tsv",
                  count_file="counts.mtx",
                  barcode_file="barcodes.tsv",
                  feature_header=None,
                  barcode_header=None,
                  feature_column=0,
                  barcode_column=0):
        """
        Creat single-omic data from files.
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
        return cls(feature_name, feature, count, barcode)

    def __getitem__(self, items):
        x, y = items
        return SingleData(self.feature_name, self.feature[x],
                          self.count[x, y], self.barcode[y])

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

    def _get_data(self):
        return {self.feature_name: self.count}, {self.feature_name: self.feature}, self.barcode


class MultiData(object):

    def __init__(self, *single_data):
        self.data = {}
        for dt in single_data:
            ct, ft, bc = dt._get_data()
            for mod in ct.keys():
                if mod not in self.data.keys():
                    self.data[mod] = {'feature': [ft[mod]], 'barcode': [bc], 'counts': [ct[mod]]}
                else:
                    self.data[mod]['feature'] += [ft[mod]]
                    self.data[mod]['barcode'] += [bc]
                    self.data[mod]['counts'] += [ct[mod]]
        for mod in self.data.keys():
            self.data[mod] = self._merge_modality(self.data[mod])

    def _merge_modality(self, dt):
        batch = [np.zeros(x.shape) + i for i, x in enumerate(dt['barcode'])]
        batch = np.concatenate(batch)
        barcode = np.concatenate(dt['barcode'])

        feature = dt['feature'][0]
        for f in dt['feature'][1:]:
            feature = np.intersect1d(feature, f)

        counts = []
        for i in range(len(dt['counts'])):
            common = np.intersect1d(feature, dt['feature'][i], return_indices=True)
            counts += [ dt['counts'][i][:, common[2]] ]
        counts = sparse.vstack(counts)

        return {'feature': feature, 'counts': counts, 'barcode': barcode, 'dataset': batch}

    def _get_data(self):
        return self.data
