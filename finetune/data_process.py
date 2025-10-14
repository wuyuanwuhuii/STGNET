from __future__ import division
from __future__ import print_function

from utils import features_construct_graph_new, spatial_construct_graph1, features_construct_graph
import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from config import Config
import sys
import anndata
import torch
import scipy.sparse as sp

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(matrix1, matrix2):
    # 将邻接矩阵转换为 numpy 数组
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    
    # 统计相同位置上数值都为1的个数
    same_count = np.sum(np.logical_and(matrix1 == 1, matrix2 == 1))
    
    # 统计 matrix2 中数值为1的总个数
    total_ones_in_matrix2 = np.sum(matrix2 == 1)
    total_ones_in_matrix1 = np.sum(matrix1 == 1)
    print("*******")
    print(same_count)
    print(total_ones_in_matrix2)
    print(total_ones_in_matrix1)
    # 计算相似度
    similarity = same_count / total_ones_in_matrix2 if total_ones_in_matrix2 > 0 else 0
    
    return similarity

def intersect_negative_graphs(graph_neg1, graph_neg2):
    """获取两个图负例的交集。
    Args:
        graph_neg1 (torch.Tensor): 第一个图的负样本邻接矩阵。
        graph_neg2 (torch.Tensor): 第二个图的负样本邻接矩阵。

    Returns:
        torch.Tensor: 两个图负例的交集邻接矩阵。
    """
    # 使用逻辑与操作找出两个图都为负例的位置
    intersection_neg = graph_neg1 * graph_neg2
    return intersection_neg


# def calculate_similarity(matrix1, matrix2):
#     # 将邻接矩阵转换为numpy数组
#     matrix1 = np.array(matrix1)
#     matrix2 = np.array(matrix2)
    
#     # 统计相同位置上数值相同的个数
#     same_count = np.sum(matrix1 == matrix2)
    
#     # 计算总矩阵大小
#     total_size = matrix1.size
    
#     # 计算相似度
#     similarity = same_count / total_size
    
#     return similarity

def generate_adj_mat_4_1(adata, include_self=False, n=6):
    # 计算皮尔逊相关系数矩阵
    correlation_matrix = cosine_similarity(adata.X)
    np.fill_diagonal(correlation_matrix, 0)  # 将对角线元素设为0，避免自身与自身的相关系数影响邻接矩阵

    # 构建邻接矩阵
    adj_mat = np.zeros_like(correlation_matrix)
    for i in range(len(adata)):
        # 找到与当前基因最相关的前n个邻居
        n_neighbors = np.argsort(correlation_matrix[i, :])[-(n+1):]  # 取后n+1个最大值，因为最大值是自身，所以取第二大到第n+1大
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        np.fill_diagonal(adj_mat, 0)  # 如果不包含自身，则将对角线元素设为0

    adj_mat = adj_mat + adj_mat.T  # 保证邻接矩阵是对称的
    adj_mat = adj_mat > 0  # 将邻接矩阵中的非零元素设为1
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


# def generate_adj_mat_3(adata, include_self=False, n=5):
#     coordinates = np.column_stack((adata.obs['x'], adata.obs['y']))
#     dist = pairwise_distances(coordinates)

#     adj_mat = np.zeros((len(adata), len(adata)))
#     for i in range(len(adata)):
#         n_neighbors = np.argsort(dist[i, :])[:n+1]
#         adj_mat[i, n_neighbors] = 1

#     if not include_self:
#         np.fill_diagonal(adj_mat, 0)

#     adj_mat = np.maximum(adj_mat, adj_mat.T)

#     graph_nei = torch.from_numpy(adj_mat)
#     graph_neg = torch.ones(adj_mat.shape[0], adj_mat.shape[0]) - graph_nei

#     sadj = sp.coo_matrix(adj_mat, dtype=np.float32)
#     sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

#     return sadj, graph_nei, graph_neg
    #return sadj


def generate_adj_mat_3(adata, include_self=False, max_distance=0.1):
    # MERFISH
    coordinates = np.column_stack((adata.obs['x'], adata.obs['y']))
    dist = pairwise_distances(coordinates)
    # xme
    # coordinates = np.column_stack((adata.obs['imagerow'], adata.obs['imagecol']))
    #OFIFISH
    #coordinates = np.column_stack((adata.obs['x_coord'], adata.obs['y_coord']))
    # # coordinates = np.column_stack((adata.obs['spatial_x'], adata.obs['spatial_y']))
    #dist = pairwise_distances(coordinates)

    # min_distance = np.min(dist[np.nonzero(dist)])
    # max_distance = np.max(dist)
    # print("********")
    # print(min_distance)
    # print(max_distance)
    
    adj_mat = np.zeros((len(adata), len(adata)))

    for i in range(len(adata)):
        # Find neighbors within max_distance
        neighbors_within_distance = np.where(dist[i] < max_distance)[0]
        adj_mat[i, neighbors_within_distance] = 1

    if not include_self:
        np.fill_diagonal(adj_mat, 0)

    adj_mat = np.maximum(adj_mat, adj_mat.T)

    graph_nei = torch.from_numpy(adj_mat)
    graph_neg = torch.ones(adj_mat.shape[0], adj_mat.shape[0]) - graph_nei

    sadj = sp.coo_matrix(adj_mat, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    return sadj, graph_nei, graph_neg
    #return sadj


def intersect_fadj(fadj1, fadj2):
    # 将两个矩阵转换为布尔型，非零为True
    bool_fadj1 = fadj1.astype(bool)
    bool_fadj2 = fadj2.astype(bool)
    intersection = bool_fadj1.multiply(bool_fadj2)
    intersection = intersection.astype(np.float32)

    return intersection


def intersect_graphs(graph1, graph2):
    # 执行逻辑与操作，找到两个邻接矩阵都为1的位置（即存在边的位置）
    intersection = torch.logical_and(graph1.bool(), graph2.bool())

    # 返回交集结果
    return intersection

def load_ST_file(highly_genes, k, radius):
    sp_data_process = anndata.read_h5ad('../pretain/dataset10_spatial_42.h5ad')
    sc_data_process = anndata.read_h5ad('../pretain/dataset10_seq_42.h5ad')
    
    tt = pd.read_csv('../Result/imputed_data/scMultiGAN_fold_1.csv',index_col=0,header=0)
    label = pd.read_csv('../pretain/demo_data/dataset10_st_label.txt',header=0,index_col= None)
    tt = tt.iloc[:42, :]
    tt = tt.T
    data = tt
    
    adata = anndata.AnnData(obs=sp_data_process.obs, var = sc_data_process.var)
  
    adata.X = data
    adata.var_names_make_unique()
    fadj = features_construct_graph(adata.X, k = 5)
 
    sadj, graph_nei, graph_neg = generate_adj_mat_3(adata, max_distance = 60)
    
    
    
    adata.obs['ground'] = label.values
    
    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    
    adata.var_names_make_unique()
    return adata

if __name__ == "__main__":
    if not os.path.exists("../generate_data/"):
        os.mkdir("../generate_data/")
    savepath = "../generate_data/"
    config_file = './config/dataset.ini'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    config = Config(config_file)
    adata = load_ST_file(config.fdim, config.k, config.radius)
    print("saving")
    adata.write(savepath + 'generate.h5ad')
    print("done")
    