
import os
from cobolt.utils import SingleData, MultiData, MultiomicDataset

dat_dir = "/Users/boyingg/Desktop/cobolt_pkg/example_data"

mop_mrna = SingleData.from_file(path=os.path.join(dat_dir, "mrna"),
                                feature_name='rna',
                                feature_file='genes.tsv')
mop_mrna.filter_features(upper_quantile=0.99, lower_quantile=0.7)
mop_atac = SingleData.from_file(path=os.path.join(dat_dir, "atac"),
                                feature_name='atac',
                                feature_file='peaks.tsv')
mop_atac.filter_features(upper_quantile=0.99, lower_quantile=0.7)
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
