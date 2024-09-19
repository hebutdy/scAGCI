import numpy as np
from sklearn.metrics import silhouette_samples


def batch_average_silhouette_width(data, labels, batch_labels):
    unique_batches = np.unique(batch_labels)
    batch_asw_scores = []

    for batch_label in unique_batches:
        batch_indices = np.where(batch_labels == batch_label)[0]
        batch_data = data[batch_indices]
        batch_labels_subset = labels[batch_indices]

        silhouette_values = silhouette_samples(batch_data, batch_labels_subset, metric='euclidean')
        batch_asw = np.mean(1 - np.abs(silhouette_values))
        batch_asw_scores.append(batch_asw)

    final_batch_asw = np.mean(1 - np.array(batch_asw_scores))

    return final_batch_asw




#batch_asw = batch_average_silhouette_width(data, labels, batch_labels)
#print("Batch Average Silhouette Width:", batch_asw)


import numpy as np
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def graph_connectivity(data, labels):
    unique_labels = np.unique(labels)
    gc_scores = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        subset_data = data[indices]

        graph = kneighbors_graph(subset_data, n_neighbors=5, mode='connectivity')
        lcc_size = len(max(nx.connected_components(nx.Graph(graph)), key=len))
        gc_score = lcc_size / len(indices)
        gc_scores.append(gc_score)

    final_gc_score = np.mean(gc_scores)

    return final_gc_score


# Example usage
gc_score = graph_connectivity(data, labels)
print("Graph Connectivity Score:", gc_score)


from sklearn.metrics import normalized_mutual_info_score, silhouette_samples
import numpy as np

def calculate_nmi(true_labels, predicted_labels):
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return nmi

def calculate_cell_type_asw(data, labels):
    unique_labels = np.unique(labels)
    silhouette_values = silhouette_samples(data, labels, metric='euclidean')
    asw = np.mean(silhouette_values)
    scaled_asw = (asw + 1) / 2
    return scaled_asw

# Example usage
true_labels = np.array([0, 1, 0, 1, 2])
predicted_labels = np.array([0, 1, 1, 0, 2])
nmi_score = calculate_nmi(true_labels, predicted_labels)
print("Normalized Mutual Information (NMI):", nmi_score)
cell_type_asw = calculate_cell_type_asw(data, labels)
print("Cell-type Average Silhouette Width (Cell-type ASW):", cell_type_asw)


import numpy as np
from scipy.stats import spearmanr

def calculate_ti_conns(pseudotime_before_integration, pseudotime_after_integration):
    spearman_corr, _ = spearmanr(pseudotime_before_integration, pseudotime_after_integration)
    ti_conns = (spearman_corr + 1) / 2
    return ti_conns

# Example usage
pseudotime_before_integration = np.array([1, 2, 3, 4, 5])
pseudotime_after_integration = np.array([2, 3, 4, 5, 6])
ti_conns_score = calculate_ti_conns(pseudotime_before_integration, pseudotime_after_integration)
print("Trajectory conservation (ti_conns) score:", ti_conns_score)
import scanpy as sc

# 读取单细胞RNA测序数据
adata = sc.read("path_to_your_file")

# 计算扩散伪时间
sc.tl.diffmap(adata)
sc.tl.dpt(adata)

# 提取伪时间值
pseudotime_before_integration = adata.obs['dpt_pseudotime']

# pseudotime_before_integration 现在包含了计算得出的伪时间值

import scanpy as sc
import anndata as ad

# 读取集成后的单细胞RNA测序数据
adata_integrated = ad.read("path_to_your_file")

# 计算扩散伪时间
sc.tl.diffmap(adata_integrated)
sc.tl.dpt(adata_integrated)

# 提取伪时间值
pseudotime_after_integration = adata_integrated.obs['dpt_pseudotime']

# pseudotime_after_integration 现在包含了计算得出的伪时间值