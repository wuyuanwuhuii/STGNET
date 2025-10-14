import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr

# 
ad_sc = sc.read("/mnt/5468e/twang/WBT/stDiff-master/datasets/sc/dataset3_seq_42.h5ad")
ad_sp = sc.read("/mnt/5468e/twang/WBT/stDiff-master/datasets/sp/dataset3_spatial_42.h5ad")

RNA_data_adata_label = ad_sc.copy()
sc.pp.normalize_total(RNA_data_adata_label)
sc.pp.log1p(RNA_data_adata_label)
sc.pp.highly_variable_genes(RNA_data_adata_label)
RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
sc.pp.scale(RNA_data_adata_label, max_value=10)
sc.tl.pca(RNA_data_adata_label)
sc.pp.neighbors(RNA_data_adata_label)
sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
ad_sc.obs['subclass_label'] = RNA_data_adata_label.obs.leiden

cell_type_column = 'subclass_label'
ad_sc.obs[cell_type_column] = ad_sc.obs[cell_type_column].astype(int)

ad_sc = ad_sc[:,~np.array([_.startswith('MT-') for _ in ad_sc.var.index])]
ad_sc = ad_sc[:,~np.array([_.startswith('mt-') for _ in ad_sc.var.index])]



RNA_data_adata_label = ad_sp.copy()
sc.pp.normalize_total(RNA_data_adata_label)
sc.pp.log1p(RNA_data_adata_label)
sc.pp.highly_variable_genes(RNA_data_adata_label)
RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
sc.pp.scale(RNA_data_adata_label, max_value=10)
sc.tl.pca(RNA_data_adata_label)
sc.pp.neighbors(RNA_data_adata_label)
sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
ad_sp.obs['subclass_label'] = RNA_data_adata_label.obs.leiden

cell_type_column = 'subclass_label'
ad_sp.obs[cell_type_column] = ad_sp.obs[cell_type_column].astype(int)


ad_sp = ad_sp[:,~np.array([_.startswith('MT-') for _ in ad_sp.var.index])]
ad_sp = ad_sp[:,~np.array([_.startswith('mt-') for _ in ad_sp.var.index])]

# 
sc.pp.normalize_total(ad_sc)
sc.pp.log1p(ad_sc)
sc.pp.normalize_total(ad_sp)
sc.pp.log1p(ad_sp)

# 
sc_expr = pd.DataFrame(ad_sc.X, index=ad_sc.obs.index, columns=ad_sc.var.index)
sp_expr = pd.DataFrame(ad_sp.X, index=ad_sp.obs.index, columns=ad_sp.var.index)

print(sc_expr.shape)  # (9234, 915)
print(sp_expr.shape)  # (645, 915)
print(np.max(ad_sc.obs['subclass_label'].values))

# 
df = pd.DataFrame(ad_sc.obs['subclass_label'].values)
df.to_csv('../dataset3/sc_dataset3_label_new.csv', index=None)

dd
correlation_matrix = np.zeros((sp_expr.shape[0], sc_expr.shape[0]))


for i in range(sp_expr.shape[0]):
    for j in range(sc_expr.shape[0]):
        correlation_matrix[i, j], _ = pearsonr(sp_expr.iloc[i, :], sc_expr.iloc[j, :])

print(correlation_matrix.shape)  # (645, 9234)


sc_labels = ad_sc.obs['subclass_label']
print(sc_labels.head())  


max_corr_indices = np.argmax(correlation_matrix, axis=1)


mapped_labels = sc_labels.iloc[max_corr_indices].values

print("*******************")
print(mapped_labels)
ad_sp.obs['mapped_sc_label'] = mapped_labels

# 
print(ad_sp.obs[['subclass_label', 'mapped_sc_label']].head())

# 
df = pd.DataFrame(ad_sp.obs['mapped_sc_label'].values, columns=['mapped_sc_label'])
df.to_csv('../dataset11/sp_dataset11_label.csv', index=None)
