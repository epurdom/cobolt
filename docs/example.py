
from cobolt.utils import SingleData, MultiData, MultiomicDataset
from cobolt.model import Cobolt
import os
import pandas as pd

dat_dir = "/Users/boyingg/Desktop/cobolt_pkg/example_data"
barcode_dir = "/Users/boyingg/Desktop/clustering_project/multiomics_clustering/rcode/20210118_method_comparison/"


mop_rna_cells = cells = pd.read_csv(
    os.path.join(barcode_dir, "rna_cells.txt"),
    header=None, sep="\t"
)[0].values.astype('str')
mop_atac_cells = pd.read_csv(
    os.path.join(barcode_dir, "atac_cells.txt"),
    header=None, sep="\t"
)[0].values.astype('str')

mop_mrna = SingleData.from_file(path=os.path.join(dat_dir, "mrna"),
                                feature_name='rna',
                                feature_file='genes.tsv')
mop_mrna.filter_features(upper_quantile=0.99, lower_quantile=0.7)
mop_mrna.filter_barcode(mop_rna_cells)
mop_atac = SingleData.from_file(path=os.path.join(dat_dir, "atac"),
                                feature_name='atac',
                                feature_file='peaks.tsv')
mop_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)
mop_atac.filter_barcode(mop_atac_cells)


snare_mrna = SingleData.from_file(path=os.path.join(dat_dir, "snare"),
                                  feature_name='rna',
                                  count_file="gene_counts.mtx",
                                  feature_file='genes.tsv')
mop_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)
snare_atac = SingleData.from_file(path=os.path.join(dat_dir, "snare"),
                                  feature_name='atac',
                                  count_file="peak_counts.mtx",
                                  feature_file='peaks.tsv')
mop_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)


all_data = MultiData(mop_mrna, mop_atac, snare_atac, snare_mrna)
multi_dt = MultiomicDataset(all_data)

model = Cobolt(dataset=multi_dt, n_latent=10)
model.train(num_epochs=5)

latent = model.get_latent()
latent_prop = model.get_topic_prop()

