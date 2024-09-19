#encoding: utf-8
import csv
import scipy.io
import os
from Clustering8Measure import Clustering8Measure,eva
from algo_qp import algo_qp,train_GAT
from scipy.io import loadmat
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.cluster import KMeans
import umap.umap_ as umap
from matplotlib import pyplot as plt
from GCN_test1 import GAT
import hdf5storage


def read_dataset( File1 = None, File2 = None, File3 = None, File4 = None, transpose = True, test_size_prop = None, state = 0,
                  format_rna = None, formar_epi = None ):
    # read single-cell multi-omics data together

    ### raw reads count of scRNA-seq data
    adata = adata1 = None

    if File1 is not None:
        if format_rna == "table":
            adata  = sc.read(File1)
        else: # 10X format
            adata  = sc.read_mtx(File1)

        if transpose:
            adata  = adata.transpose()

    ##$ the binarization data for scEpigenomics file
    if File2 is not None:
        if formar_epi == "table":
            adata1  = sc.read( File2 )
        else  :# 10X format
            adata1  = sc.read_mtx(File2)

        if transpose:
            adata1  = adata1.transpose()

    ### File3 and File4 for cell group information of scRNA-seq and scEpigenomics data
    label_ground_truth = []
    label_ground_truth1 = []

    if state == 0 :
        if File3 is not None:
            Data2  = pd.read_csv( File3, header=0, index_col=0 )
            label_ground_truth =  Data2['Group'].values

        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_csv( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Group'].values

        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    elif state == 3:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['Group'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Group'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    else:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    # split datasets into training and testing sets
    if test_size_prop > 0 :
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs),
                                               test_size = test_size_prop,
                                               random_state = 200)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx]  = 'test'
        adata.obs['split']  = spl.values

        if File2 is not None:
            adata1.obs['split'] = spl.values
    else:
        train_idx, test_idx = list(range( adata.n_obs )), list(range( adata.n_obs ))
        spl = pd.Series(['train'] * adata.n_obs)
        adata.obs['split']       = spl.values

        if File2 is not None:
            adata1.obs['split']  = spl.values

    adata.obs['split'] = adata.obs['split'].astype('category')
    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')

    if File2 is not None:
        adata1.obs['split'] = adata1.obs['split'].astype('category')
        adata1.obs['Group'] = label_ground_truth
        adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    ### here, adata with cells * features
    return adata, adata1, train_idx, test_idx, label_ground_truth, label_ground_truth1



def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.sum(adata.X, axis=1))
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata



if __name__ == "__main__":
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    #data_root  =  './datasets/pbmc'

    #X1 = os.path.join(data_root, 'rna_data_2874cell_1500genes.txt')

    #X2 = os.path.join(data_root, 'atac_data_2874cell_11063loci_binary.txt')
    #x3 = os.path.join(data_root, 'pbmc_3k_cell_cluster_2874.txt')  # cell type information #state=1
    #data_root = './datasets/snare'

    #X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    #X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    #x3 = os.path.join(data_root, 'cell_metadata.txt')  # cell type information  #state=1
    data_root='./datasets/Mouse_skin'
    X1 = os.path.join(data_root, 'Skin_2000_rna_34774cells.txt')
    X2 = os.path.join(data_root, 'Skin_35943_atac_34774cells_binary.txt')
    x3 = os.path.join(data_root, 'GSM4156597_skin_cluster.txt')  # cell type information
    # # adata: scRNA-seq with samples x genes
    # # adata: scATAC    with samples x peaks
    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=3, format_rna="table",
                                                                          formar_epi="table")

    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    print('===== Normalize RNA =====')

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize ATAC=====')
    #for RNA
    x_scRNA = x1.X  #1047cell 500gene
    # pca = PCA(n_components=300)
    # x_scRNA = pca.fit_transform(x_scRNA)

    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values

    x_scRNA = torch.from_numpy(x_scRNA)
    x_scRNAraw = torch.from_numpy(x_scRNAraw)
    x_scRNA_size_factor = torch.from_numpy(x_scRNA_size_factor)

    N1, M1 = np.shape(x_scRNA)

    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    # For scATAC
    x_scATAC = x2.X    #1047cell 7136peaks
    # pca = PCA(n_components=3000)
    # x_scATAC = pca.fit_transform(x_scATAC)

    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = torch.from_numpy(x_scATAC)
    x_scATACraw = torch.from_numpy(x_scATACraw)
    x_scATAC_size_factor = torch.from_numpy(x_scATAC_size_factor)

    N2, M2 = np.shape(x_scATAC)
    merged_data = torch.cat((x_scRNA, x_scATAC), dim=1)
    scRNA = x_scRNA.numpy()  # 将tensor转换为numpy数组
    scATAC = x_scATAC.numpy()
    merge= merged_data.numpy()
    #以cell形式存储
    #singlecell=[scRNA,scATAC,merge]
    singlecell = [scRNA, scATAC,merge]
    singlecell_=np.empty((len(singlecell),1),dtype=np.object)
    singlecell_[0][0]=singlecell[0]
    singlecell_[1][0] = singlecell[1]
    singlecell_[2][0] = singlecell[2]
    data = {'X':singlecell_, 'Y': label_ground_truth}
    # 保存为mat文件
    #hdf5storage.savemat('skin.mat', data,format='7.3',matlab_compatible=True)

    #data = hdf5storage.loadmat('skin.mat')
    anchor_rate = [5,10,20,30,40,50,60,70]  # 锚点选择
    #d_rate = []

    for dsi in np.array(range(0, 1, 1)).reshape(-1):
        X = data['X']
        Y = data['Y']
        Y=Y.reshape(-1,1)
        k = np.unique(Y).shape[0]  # Y的类数
        n = Y.shape[0]
        numview = X.shape[0]

         # 标准化，使得每个特征均值为0，方差为1
        mean = 0
        std = 0
        for i in range(numview):
            mean = np.mean(X[i][0].T, axis=1, keepdims=True)
            std = np.std(X[i][0].T, axis=1, keepdims=True)
            X[i][0]= (X[i][0].T - mean) / std

        for ichor in range(len(anchor_rate)):
            #for id in range(len(d_rate)):
            A, W, Z, iter, obj, alpha, label,output,adj_tensor = algo_qp(X, Y, k, anchor_rate[ichor] * k)
            print(Z.shape)
            # 锚点学习输出
            print(anchor_rate[ichor], iter)
            ACC, nmi, Purity, Fscore, Precision, Recall, ARI = Clustering8Measure(Y, label)
            scaled_asw, final_gc_score=eva(X[2][0].T,label)
            print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI,scaled_asw,final_gc_score)
    #Z是共性图 维度（m,n)
    #层次化图注意力








