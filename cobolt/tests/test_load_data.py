
import pytest
import os
import numpy as np
import scipy
from cobolt.utils import SingleData, MultiData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_test_data():
    ja = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_a"),
        dataset_name="joint", feature_name="a")
    jb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "joint_b"),
        dataset_name="joint", feature_name="b")
    sa = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_a"),
        dataset_name="single_a", feature_name="a")
    sb = SingleData.from_file(
        path=os.path.join(THIS_DIR, "test_data", "single_b"),
        dataset_name="single_b", feature_name="b")
    return ja, jb, sa, sb


class TestSingleData:
    def test_construction(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        count, feature, barcode = ja.get_data()
        assert ja.get_dataset_name() == "joint"
        assert count[feature_name].shape == (100, 100)
        assert feature[feature_name].shape == (100, )
        assert barcode.shape == (100, )
        assert isinstance(feature[feature_name], np.ndarray)
        assert isinstance(barcode, np.ndarray)
        assert isinstance(count[feature_name], scipy.sparse.csr.csr_matrix)

    def test_filter_features(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_features(min_count=2, min_cell=1)
        count, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (100, 46)
        assert (count[feature_name].sum(axis=0) > 2).all()
        assert ((count[feature_name] != 0).sum(axis=0) > 1).all()
        assert feature[feature_name].shape == (46, )
        assert barcode.shape == (100, )

    def test_filter_cells(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_cells(min_count=2, min_feature=1)
        count, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (85, 100)
        assert (count[feature_name].sum(axis=1) > 2).all()
        assert ((count[feature_name] != 0).sum(axis=1) > 1).all()
        assert feature[feature_name].shape == (100, )
        assert barcode.shape == (82, )

    def test_filter_barcode(self):
        feature_name = "a"
        ja, jb, sa, sb = load_test_data()
        ja.filter_barcode(cells=[
            '09A_CAGCCCCGCCTT',
            '09A_CGCCTACCATGA'
        ])
        count, feature, barcode = ja.get_data()
        assert count[feature_name].shape == (2, 100)
        assert feature[feature_name].shape == (100, )
        assert barcode.shape == (2, )


class TestMultiData:
    def test_construction(self):
        ja, jb, sa, sb = load_test_data()
        multi = MultiData(ja, jb, sa, sb).get_data()
        assert list(multi.keys()) == ['a', 'b']


class TestDataset:
    def test_construction(self):
        pass


