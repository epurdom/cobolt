
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
import numpy as np


class ClusterUtil:

    def __init__(self, k=20, key=0):
        self.key=key
        self.k=k

    def check_version(self, k, key):
        return self.k == k and self.key == key

    def fit(self, latent):
        self.latent = latent
        self.snn_mat = snn_from_data(latent, self.k)
        self.graph = graph_from_snn(self.snn_mat)
        self.cluster = {}

    def run_louvain(self):
        raise NotImplementedError

    def run_leiden(self, resolution=1, seed=0):
        kwargs = {'weights': np.array(self.graph.es['weight']).astype(np.float64),
                  'resolution_parameter': resolution}
        partition = leidenalg.find_partition(
            self.graph,
            partition_type=leidenalg.RBConfigurationVertexPartition,
            seed=seed,
            **kwargs
        )
        self.cluster['leiden_{:.3f}'.format(1)] = partition.membership

    def get_clusters(self, algo="leiden", resolution=1):
        return self.cluster['{}_{:.3f}'.format(algo, resolution)]


def snn_from_data(latent, k):
    knn_mat = kneighbors_graph(latent, k, mode='connectivity', include_self=False)
    snn_mat = knn_mat.dot(knn_mat.T)
    snn_mat.data[:] = snn_mat.data / (k + k - snn_mat.data)
    snn_mat.setdiag(0)
    return snn_mat


def graph_from_snn(snn_mat):
    snn_mat.eliminate_zeros()
    graph = ig.Graph(n=snn_mat.shape[0],
                     edges=list(zip(*snn_mat.nonzero())),
                     edge_attrs={'weight': snn_mat.data},
                     directed=True)
    return graph
