
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os

dat_dir = "../../example_data"

snare_mrna = SingleData.from_file(path=os.path.join(dat_dir, "snare"),
                                  dataset_name="SNARE-seq",
                                  feature_name="GeneExpr",
                                  count_file="gene_counts.mtx",
                                  feature_file="genes.tsv")
snare_mrna.filter_features(upper_quantile=0.99, lower_quantile=0.7)
snare_atac = SingleData.from_file(path=os.path.join(dat_dir, "snare"),
                                  dataset_name="SNARE-seq",
                                  feature_name="ChromAccess",
                                  count_file="peak_counts.mtx",
                                  feature_file="peaks.tsv")
snare_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)


mop_mrna = SingleData.from_file(path=os.path.join(dat_dir, "mrna"),
                                dataset_name="mRNA",
                                feature_name="GeneExpr",
                                feature_file="genes.tsv")
mop_mrna.filter_features(upper_quantile=0.99, lower_quantile=0.7)

mop_atac = SingleData.from_file(path=os.path.join(dat_dir, "atac"),
                                dataset_name="ATAC",
                                feature_name="ChromAccess",
                                feature_file="peaks.tsv")
mop_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)

multi_dt = MultiomicDataset.from_singledata(
    mop_mrna, mop_atac, snare_atac, snare_mrna)
print(multi_dt)

model = Cobolt(dataset=multi_dt, n_latent=10)
model.train(num_epochs=5)

model.calc_all_latent()

model.clustering(algo="louvain")
c1 = model.get_clusters("louvain")
model.clustering(algo="leiden", resolution=0.5)
c2 = model.get_clusters("leiden", 0.5)

model.scatter_plot(reduc="UMAP", correction=True)
