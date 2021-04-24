
import pytest
import numpy as np
from cobolt.model.clustering import ClusterUtil


class TestClusterUtil:
    def test_check_version(self):
        obj = ClusterUtil(20, 30)
        assert not obj.check_version(10, 30)
        assert not obj.check_version(20, 10)
        assert not obj.check_version(0, 0)

    def test_fit(self):
        dt = np.random.sample((100, 200))
        obj = ClusterUtil(20, 30)
        obj.fit(dt)
        assert obj.snn_mat.max() <= 1
        assert obj.snn_mat.min() >= 0
        assert obj.snn_mat.shape == (100, 100)
        assert obj.latent.shape == (100, 200)
        assert len(obj.graph.vs) == 100
        assert obj.graph.es['weight'] is not None

    def test_run_louvain(self, capfd):
        dt = np.random.sample((100, 200))
        obj = ClusterUtil(20, 30)
        obj.fit(dt)
        res1 = obj.get_clusters("louvain")
        out, err = capfd.readouterr()
        assert out.startswith("Running")
        assert len(res1) == 100
        obj.run_louvain()
        out, err = capfd.readouterr()
        assert out.startswith("Clustering results")
        res2 = obj.get_clusters("louvain")
        assert res1 is res2
        obj.run_louvain(overwrite=True)
        out, err = capfd.readouterr()
        assert out.startswith("Running")
        res3 = obj.get_clusters("louvain")
        assert res1 is not res3

    def test_run_leiden(self, capfd):
        dt = np.random.sample((100, 200))
        obj = ClusterUtil(20, 30)
        obj.fit(dt)
        res1 = obj.get_clusters("leiden", resolution=1)
        out, err = capfd.readouterr()
        assert out.startswith("Running")
        assert len(res1) == 100
        res2 = obj.get_clusters("leiden", resolution=0.5)
        out, err = capfd.readouterr()
        assert out.startswith("Running")
        assert res1 is not res2
        obj.run_leiden(resolution=1, overwrite=False)
        out, err = capfd.readouterr()
        assert out.startswith("Clustering results")
        res3 = obj.get_clusters("leiden", resolution=1)
        assert res1 is res3
        obj.run_leiden(resolution=1, overwrite=True)
        out, err = capfd.readouterr()
        assert out.startswith("Running")
        res4 = obj.get_clusters("leiden", resolution=1)
        assert res1 is not res4
