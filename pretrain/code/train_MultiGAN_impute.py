from __future__ import print_function, division
import argparse
from pathlib import Path
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import Generator
import Discriminator
#from scMultiGAN_impute_1 import scMultiGAN_impute
#from scMultiGAN_impute import scMultiGAN_impute
from MultiGAN_impute import scMultiGAN_impute
import imputer_old
from utiles import MyDataset,ToTensor,one_hot,mask_data,mkdir, MyDataset_sc,ToTensor_sc,MyDataset_old
from torch.nn.parallel import DataParallel
import scanpy as sc
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"s

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default= 500)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--alpha', type=float, default=.1) 
parser.add_argument('--gamma', type=float, default=0)
parser.add_argument('--beta', type=float, default=.1)

parser.add_argument('--pretrain', type=str,default='../datasets12_sc_one/model_impute/0499.pth',help='path of scMultiGAN\'s pretrain model')
parser.add_argument('--lambd', type=float, default=10)
parser.add_argument('--n-critic', type=int, default=5) 
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--d_file', type=str, default='../demo_data/train_fold_1_dataset12.csv', help='path of data file')
parser.add_argument('--gene_mask', type=str, default="../demo_data/zero_genes_fold_1.csv", help='path of cls file')
parser.add_argument('--c_file', type=str, default='../demo_data/dataset10_st_label.txt', help='path of cls file')
parser.add_argument('--coords_file', type=str, default='../demo_data/dataset10_coords.csv', help='path of cls file')
parser.add_argument('--sc_data', type=str, default='../demo_data/dataset10_seq_42.h5ad', help='path of cls file')

parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--img_size', type=int, default=7, help='size of each image dimension')
parser.add_argument('--imputeronly', action='store_false',default = True)
parser.add_argument('--imputer_model', type=str,default="/mnt/5468e/twang/WBT/scMultiGAN-main/scMultiGAN-main/dataset12_st_one/model_impute/0499_12_5.pth",help='path of pretrain model')
parser.add_argument('--output_dir',type=str,default='../Result/',help='path of the project')
parser.add_argument('--checkpoint',type=str,default=None,help='path of the output_checkpoint')
parser.add_argument('--lr',type=int,default=1e-5,help='Learnsing rate')  # 1e-5  -> 1e-4
parser.add_argument('--ncls', type=int, default=14,help='number of clusters')   # changed
parser.add_argument('--num_workers',type=int,default=1,help="cores for loading data")
opt = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device('cuda:3' if cuda else 'cpu')
#device = 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
imputer_1 = imputer_old.UNetImputer_sc(channels=opt.channels,img_size=opt.img_size,ncls=opt.ncls).to(device)
imputer_2 = imputer_old.UNetImputer_st(channels=opt.channels,img_size=opt.img_size,ncls=opt.ncls).to(device)
impu_critic_st = Discriminator.ConvCritic_st(img_size=opt.img_size,ncls=opt.ncls,channels=opt.channels).to(device)
impu_critic_sc = Discriminator.ConvCritic(img_size=opt.img_size,ncls=opt.ncls,channels=opt.channels).to(device)
print("\n Using CUDA: {}\n".format(cuda)+"********************************")

ad_res = sc.read(opt.sc_data)
genes_res = ad_res.var.index.tolist()
all_genes = genes_res
with open(opt.gene_mask, "r") as file:
    excluded_genes = [line.strip() for line in file.readlines()]
print(excluded_genes)

gene_selection_mask = np.array([gene in excluded_genes for gene in all_genes])
gene_selection_mask = ~gene_selection_mask
gene_selection_mask = gene_selection_mask.astype(int)
selected_gene_list = np.array(all_genes)[gene_selection_mask]
opt.gene_mask = gene_selection_mask



print("Training imputer network: {}".format(opt.imputeronly))
#output_dir = Path('../breast_cancer')
if(opt.imputeronly):
    print("Reading data from disk\n" +"********************************")
    transformed_dataset = MyDataset(opt=opt,transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,shuffle=True, num_workers=opt.num_workers,drop_last=True)
    print("expression data path is: {}\n".format(opt.d_file))
    print("mask data path is: {}\n".format(opt.c_file))
    print("Reading data finished, start training\n"+"********************************")
    data_gene = Generator.ConvDataGenerator(img_size=opt.img_size,ncls=opt.ncls,latent_dim=opt.latent_dim,channels=opt.channels).to(device)
    #scMultiGAN_impute(opt, data_gene, imputer_1, impu_critic_st, impu_critic_sc, dataloader)   # change
    #scMultiGAN_impute(opt, data_gene, imputer_1, impu_critic_sc, dataloader)
    scMultiGAN_impute(opt, imputer_2, imputer_1, impu_critic_st, impu_critic_sc, dataloader)   # change



for param in imputer_2.parameters():
    param.requires_grad = False
print("********************************")
print("Loading trained imputer_model\n"+"********************************")
imputer_model = torch.load(Path(opt.imputer_model))
imputer_2.load_state_dict(imputer_model['imputer'])

print("Starting reading expression data and mask data from disk\n"+"********************************")
transformed_dataset = MyDataset(opt=opt,transform=transforms.Compose([ToTensor()]))
batch_sampler = BatchSampler(torch.utils.data.SequentialSampler(transformed_dataset),opt.batch_size,drop_last=False)
dataloader = DataLoader(transformed_dataset, batch_sampler=batch_sampler, shuffle=False, num_workers=42)
print("expression data path is: {}".format(opt.d_file))
print("mask data path is: {}\n".format(opt.c_file))
print("Reading data finished, start impute expression data\n"+"********************************")
print(dataloader)

result = pd.DataFrame()
torch.manual_seed(123)
for j,real_sample in enumerate(dataloader):
    real_data = real_sample['real_data'].to(device)
    real_mask = real_sample['real_mask'].to(device)
    cord = real_sample['coords'].to(device)
    
    label = real_sample['label'].to(device)
    label_hot = one_hot((label).type(torch.LongTensor), opt.ncls).type(Tensor).to(device)
    impu_noise = torch.FloatTensor(real_data.shape[0], 1, opt.img_size, opt.img_size).to(device)
    impu_noise.normal_()
    impute_data = imputer_2(real_data, real_mask, impu_noise,label_hot,cord) # change
    masked_imputed_data = mask_data(real_data, real_mask, impute_data)
    #masked_imputed_data = impute_data
    masked_imputed_data = masked_imputed_data.reshape((real_data.shape[0],opt.img_size**2)).T
    masked_imputed_data = masked_imputed_data.detach().cpu().numpy()
    result = pd.concat([result,pd.DataFrame(masked_imputed_data)],ignore_index=True,axis=1)
print("Imputering data finished, start writing to the disk\n"+"********************************")
mkdir(Path(opt.output_dir)/'imputed_data')
result.to_csv(Path(os.path.join(opt.output_dir,'imputed_data',"scMultiGAN_fold_1.csv")) ,index=False)
print("Finished\n"+"********************************")
