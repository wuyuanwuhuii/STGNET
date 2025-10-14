from torch.autograd import grad
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F



class CriticUpdater:
    def __init__(self, critic, critic_optimizer, eps, ones, lambd=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.lambd = lambd

    def __call__(self, real, fake,label_hot):
        real = real.detach()
        fake = fake.detach()
        label_hot = label_hot.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp,label_hot), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.lambd
        w_dist = self.critic(fake,label_hot).mean() - self.critic(real,label_hot).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()
        

class CriticUpdater_st:
    def __init__(self, critic, critic_optimizer, eps, ones, lambd=10, alpha=0.1):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.lambd = lambd
        self.alpha = alpha  # Weight for the reconstruction loss

    def __call__(self, real, fake,mask, label_hot, cord):
        real = real.detach()
        fake = fake.detach()
        label_hot = label_hot.detach()
        cord = cord.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp,label_hot, cord), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.lambd
        w_dist = self.critic(fake,label_hot, cord).mean() - self.critic(real,label_hot, cord).mean()
        
        # KL divergence loss for unmasked genes
        #rec_loss = F.kl_div(F.log_softmax(fake, dim=-1), F.softmax(real, dim=-1), reduction='batchmean') * mask.mean()
        
        # Reconstruction loss for unmasked genes
        rec_loss = torch.mean((fake - real) ** 2)
        
        loss = w_dist + grad_penalty + 100 * rec_loss
        loss_2 = 100 * rec_loss
        #print("****************")
        #print(loss.detach())
        #print(loss_2.detach())
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()        




def mask_norm(diff, mask):
    dim = 1, 2, 3
    return ((diff * mask).sum(dim) / mask.sum(dim)).mean()

def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau

def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return

def one_hot(batch,depth):
    ones = torch.eye(depth)
    return ones.index_select(0,batch)

                
class ToTensor_sc(object):
    def __call__(self, sample):
        data,label,mask = sample['real_data'], sample['label'], sample['real_mask']
        data = data.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {'real_data': torch.from_numpy(data).to(torch.float32),
                'label': torch.from_numpy(label),
                'real_mask': torch.from_numpy(mask).to(torch.float32)
                }

class ToTensor(object):
    def __call__(self, sample):
        data,label,mask,coords = sample['real_data'], sample['label'], sample['real_mask'], sample['coords']
        data = data.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {'real_data': torch.from_numpy(data).to(torch.float32),
                'label': torch.from_numpy(label),
                'real_mask': torch.from_numpy(mask).to(torch.float32),
                'coords': torch.from_numpy(coords).to(torch.float32)  # 
                }
 
class MyDataset(Dataset):
    def __init__(self, opt,transform=None):
        self.data = pd.read_csv(opt.d_file, header=0, index_col=0,sep=",")
        d = pd.read_csv(opt.c_file, header=0, index_col=False)  
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes  
        
        self.coords = pd.read_csv(opt.coords_file, header=0, index_col=False, usecols=['x', 'y'])
        
        #self.mask = opt.gene_mask
        self.gene_mask = self.expand_gene_mask(opt.gene_mask, opt.img_size)   #1
        
        self.transform = transform
        self.fig_h = opt.img_size  
    def __len__(self):
        return len(self.data_cls)
    
    def expand_gene_mask(self, gene_mask, img_size):
        expanded_mask = np.zeros((img_size * img_size,), dtype=int)
        expanded_mask[:len(gene_mask)] = gene_mask
        return expanded_mask.reshape((img_size, img_size, 1))

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'float')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        
        mask = self.gene_mask.astype('float')     #2
        
        #print(sum(mask))
        
        #mask = np.where(data>0,1,0).astype('float')
        #mask = self.mask
        coords = self.coords.iloc[idx].values.astype('float')
        
        sample = {'real_data': data, 'label': label, 'real_mask':mask, 'coords':coords}
        #sample = {'real_data': data, 'label': label, 'real_mask':mask}
        #print(sample.keys()) 
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class MyDataset_old(Dataset):
    def __init__(self, opt,transform=None):
        self.data = pd.read_csv(opt.d_file, header=0, index_col=0,sep=",")
        d = pd.read_csv(opt.c_file, header=0, index_col=False)  
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes 
        
        self.coords = pd.read_csv(opt.coords_file, header=0, index_col=False, usecols=['x', 'y']) 
        #self.gene_mask = self.expand_gene_mask(opt.gene_mask, opt.img_size)   #1
        self.transform = transform
        self.fig_h = opt.img_size  
    def __len__(self):
        return len(self.data_cls)
        
    def expand_gene_mask(self, gene_mask, img_size):
        expanded_mask = np.zeros((img_size * img_size,), dtype=int)
        expanded_mask[:len(gene_mask)] = gene_mask
        return expanded_mask.reshape((img_size, img_size, 1))

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'float')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        mask = np.where(data>0,1,0).astype('float')
        coords = self.coords.iloc[idx].values.astype('float')
        sample = {'real_data': data, 'label': label, 'real_mask':mask, 'coords':coords}
        if self.transform:
            sample = self.transform(sample)
        return sample

class MyDataset_sc(Dataset):
    def __init__(self, opt,transform=None):
        self.data = pd.read_csv(opt.d_file, header=0, index_col=0,sep=",")
        d = pd.read_csv(opt.c_file, header=0, index_col=False)  
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes
        #self.mask = opt.gene_mask  
        
        
        self.transform = transform
        self.fig_h = opt.img_size  
    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'float')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        mask = np.where(data>0,1,0).astype('float')
        

        sample = {'real_data': data, 'label': label, 'real_mask':mask}
        #print(sample.keys()) 
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self,data_loss,out_images,real_images):
        adversarial_loss = data_loss
        image_loss = self.mse_loss(out_images, real_images)
        return image_loss + 0.1 * adversarial_loss