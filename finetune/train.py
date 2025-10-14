from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scanpy as sc
os.chdir(sys.path[0])
import torch.nn.functional as F
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from scipy.stats import nbinom
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
def SPCC_2(raw, impute):
    print(type(raw))
    print(type(impute))
    spearman_values = []
    for i in range(raw.shape[1]):
        # 使用numpy的nan_to_num来处理NaN，将NaN值替换为一个非常小的数（1e-20）
        raw_col = np.nan_to_num(raw[:, i], nan=1e-20)
        impute_col = np.nan_to_num(impute[:, i], nan=1e-20)
        
        # 计算Spearman相关系数
        spearman_value, _ = spearmanr(raw_col, impute_col)
        spearman_values.append(spearman_value)
    median_spearman = np.median(spearman_values)
    return median_spearman

def SSIM_2(raw, impute):
    ssim_values = []
    for i in range(raw.shape[1]):
        # 使用numpy的nan_to_num来处理NaN，将NaN值替换为一个非常小的数（1e-20）
        raw_col = np.nan_to_num(raw[:, i], nan=1e-20)
        impute_col = np.nan_to_num(impute[:, i], nan=1e-20)
        
        # 计算数据的最大值和最小值
        data_range = max(raw_col.max() - raw_col.min(), impute_col.max() - impute_col.min())
        
        # 计算SSIM值，指定data_range
        ssim_value = ssim(raw_col, impute_col, data_range=data_range)
        ssim_values.append(ssim_value)
        
    median_ssim = np.median(ssim_values)
    return median_ssim



def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata


def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

def scale_max(arr):
    arr = np.array(arr)
    max_vals = np.max(arr, axis=0)  # 找出每列的最大值
    max_vals[max_vals == 0] = 1  # 防止最大值为 0 的除零错误
    return arr / max_vals

def SSIM_3(raw, impute):
    
    raw = scale_max(raw)
    impute = scale_max(impute)
    ssim_values = []
    data_range = max(raw.max(), impute.max())  # 计算数据的最大范围
    
    for i in range(raw.shape[1]):  # 遍历每个基因列
        raw_col = np.nan_to_num(raw[:, i], nan=1e-20)  # 处理 NaN 值
        impute_col = np.nan_to_num(impute[:, i], nan=1e-20)  # 处理 NaN 值
        
        # 计算 SSIM 值
        ssim_value = cal_ssim(raw_col.reshape(-1, 1), impute_col.reshape(-1, 1), data_range)
        ssim_values.append(ssim_value)

    # 使用基因名作为索引，返回 Pandas Series
    median_ssim = np.sum(ssim_values)/len(ssim_values)
    return median_ssim


def SPCC(raw, impute):
    # 确保输入是 NumPy 数组
    raw = np.asarray(raw)
    impute = np.asarray(impute)
    
    # 检查维度
    if raw.shape != impute.shape:
        #print(f"Dimension mismatch: raw shape is {raw.shape}, impute shape is {impute.shape}")
        return None

    spearman_values = []
    num_skipped = 0
    for i in range(raw.shape[1]):
        raw_col = raw[:, i]
        impute_col = impute[:, i]
        
        # 检查 NaN 和 Inf
        if np.isnan(raw_col).any() or np.isnan(impute_col).any():
            num_skipped += 1
            continue
        if np.isinf(raw_col).any() or np.isinf(impute_col).any():
            num_skipped += 1
            continue

        # 检查常数列
        if np.std(raw_col) == 0 or np.std(impute_col) == 0:
            num_skipped += 1
            continue

        # 计算 Spearman 相关系数
        spearman_value, _ = spearmanr(raw_col, impute_col)
        if not np.isnan(spearman_value):
            spearman_values.append(spearman_value)
        else:
            num_skipped += 1

    if len(spearman_values) == 0:
        print('No valid Spearman correlations could be computed.')
        return np.nan

    mean_spearman = np.mean(spearman_values)
    #print(f'Spearman correlations computed for {len(spearman_values)} columns, skipped {num_skipped} columns.')
    return mean_spearman


def SPCC_2(raw, impute):
    # 检查输入的维度是否匹配
    if raw.shape != impute.shape:
        #print(f"Dimension mismatch: raw shape is {raw.shape}, impute shape is {impute.shape}")
        return None
    
    if hasattr(raw, 'toarray'):  # 如果是 SciPy 稀疏矩阵
        raw = raw.toarray()
    if hasattr(impute, 'toarray'):
        impute = impute.toarray()

    spearman_values = []
    for i in range(raw.shape[1]):
        raw_col = np.nan_to_num(raw[:, i], nan=1e-20)
        impute_col = np.nan_to_num(impute[:, i], nan=1e-20)

        # 计算每一列的 Spearman 相关系数
        spearman_value, _ = spearmanr(raw_col, impute_col)
        spearman_values.append(spearman_value)

    # 返回所有列的 Spearman 相关系数的平均值
    mean_spearman = np.mean(spearman_values)
    return mean_spearman



def load_data(dataset):
    print("load data:")
    path = "../generate_data/generate.h5ad"
    sp_data_process = sc.read_h5ad('./pretain/demo_data/dataset10_spatial_42.h5ad')
    sp_data_process.X = sp_data_process.X.toarray()
    sc.pp.normalize_total(sp_data_process, target_sum=1e4)
    sc.pp.log1p(sp_data_process)
    print(np.max(sp_data_process.X))
    sp_data_process = scale(sp_data_process)
    
    adata = sc.read_h5ad(path)
    
    genes = pd.read_csv('./pretain/demo_data/zero_genes.csv',index_col=0,header=0) ## 进行插补的基因
    gene_values = genes.values.flatten()  # 将二维数组展开为一维数组
    
    #gene_mask = gene_in.adata.var.index.isin(gene_values_bool)
    gene_mask_in_adata = adata.var.index.isin(gene_values)

    # 对 gene_mask_in_adata 取反
    gene_mask = ~gene_mask_in_adata
    features = torch.FloatTensor(adata.X)
    ground_features = sp_data_process.X#.toarray()
    labels = adata.obs['ground']
    print(labels)
    print(labels)
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
   
    print("done")
    return adata, features,labels, nfadj, nsadj, graph_nei, graph_neg, ground_features, gene_mask



def kl_divergence_2(real_features, predicted_mean):
    if isinstance(real_features, np.ndarray):
        real_features = torch.tensor(real_features, dtype=torch.float32)
    if isinstance(predicted_mean, np.ndarray):
        predicted_mean = torch.tensor(predicted_mean, dtype=torch.float32)
     # 将稀疏矩阵转换为密集矩阵
    if isinstance(predicted_mean, torch.sparse.FloatTensor):
        predicted_mean = predicted_mean.to_dense()
    if isinstance(real_features, torch.sparse.FloatTensor):
        real_features = real_features.to_dense()
    log_predicted = F.log_softmax(predicted_mean, dim=1)  # 转换成对数概率
    target_prob = F.softmax(real_features, dim=1)         # 真实特征转换为概率
    kl_loss = F.kl_div(log_predicted, target_prob, reduction='batchmean')  # 计算KL散度
    return kl_loss

def kl_divergence(real_features, predicted_mean):
    # 如果 real_features 或 predicted_mean 是稀疏矩阵，转换为密集矩阵
    if hasattr(predicted_mean, 'toarray'):  # 如果是 SciPy 稀疏矩阵
        predicted_mean = predicted_mean.toarray()
    if hasattr(real_features, 'toarray'):
        real_features = real_features.toarray()

    # 如果是 NumPy 数组，转换为 PyTorch 张量
    if isinstance(predicted_mean, np.ndarray):
        predicted_mean = torch.tensor(predicted_mean, dtype=torch.float32)
    if isinstance(real_features, np.ndarray):
        real_features = torch.tensor(real_features, dtype=torch.float32)

    # 确保稀疏矩阵转换为密集矩阵
    if predicted_mean.is_sparse:
        predicted_mean = predicted_mean.to_dense()
    if real_features.is_sparse:
        real_features = real_features.to_dense()

    # 使用 log_softmax 和 softmax 计算 KL 散度
    log_predicted = F.log_softmax(predicted_mean, dim=1)  # 转换成对数概率
    target_prob = F.softmax(real_features, dim=1)         # 真实特征转换为概率
    kl_loss = F.kl_div(log_predicted, target_prob, reduction='batchmean')  # 计算KL散度
    return kl_loss


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
   
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
   
    kl_loss = kl_divergence(mean[:,~gene_mask], ground_features)

    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss + kl_loss * config.beta 
    
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, pi,disp, mean, zinb_loss * config.alpha, reg_loss * config.gamma, con_loss * 10 ,kl_loss * config.beta, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['mop']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/dataset.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg, ground_features, gene_mask = load_data(dataset)
        print(adata)

        plt.rcParams["figure.figsize"] = (3, 3) 
        savepath = './result/' + dataset + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        title = "Manual annotation (slice #" + dataset + ")"

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = 100
        config.epochs = config.epochs + 1

        if cuda:
            features = features.cuda()
            gene_mask = gene_mask.cuda()
            ground_features = ground_features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed( config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma, ' ', config.n, ' ',config.class_num)
        model = Spatial_MGCN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        min_loss = 1e5
        max_loss_spcc = 0
        max_loss_mse = 1e5
        best_mse = 1e4

        for epoch in range(config.epochs):
            emb, pi,disp, mean, zinb_loss, reg_loss, con_loss, kl_loss, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),' kl_loss = {:.2f}'.format(kl_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            
            kl_loss_3 = SPCC_2(mean[:,~gene_mask], ground_features)
           
            if kl_loss_3 > max_loss_spcc: 
                i = epoch
                #ari_max = ari_res
                epoch_max = epoch
                #idx_max = idx
                mean_max = mean
                emb_max = emb
                pi_max = pi
                disp_max = disp
                max_loss_spcc = kl_loss_3
                max_loss_mse = kl_loss_3
                min_loss = total_loss
                #best_mse = mse

        print(dataset, ' ', max_loss_spcc , ' ', i)
        title = 'Spatial-MGCN: ARI={:.2f}'.format(ari_max)
        adata.obsm['mean'] = mean_max

        pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_emb.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        df = pd.DataFrame(adata.layers['mean'])
        df.to_csv(savepath + 'ours_generate_fold_1.csv',header=adata.var_names.values)



