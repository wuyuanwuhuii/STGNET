import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        # self.DispAct = lambda x: torch.clamp(F.softplus(x), 0, 1)
        # self.MeanAct = lambda x: torch.clamp(torch.exp(x),0,1)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.sigmoid(x)
        #self.MeanAct = lambda x: torch.sigmoid(x)
        #self.MeanAct = torch.sigmoid
        # self.DispAct = lambda x: torch.clamp(F.softplus(x), 0, 1)
        # self.MeanAct = lambda x: torch.sigmoid(x)
    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
     
        return [pi, disp, mean]





class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta



class Spatial_MGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(Spatial_MGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        
        self.dropout = dropout
        
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )
        
        # 添加可学习的超参数a和b
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        self.c = nn.Parameter(torch.rand(1))

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        com1 = self.CGCN(x, sadj)  # Co_GCN
        com2 = self.CGCN(x, fadj)  # Co_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN

        # 使用加权和替换原来的平均
        #com_comb = 0.3 * com1 + 0.7 * com2
        #com_comb = self.a * com1 + self.a * com2
        com_comb = (com1 + com2)/2
        #emb = emb1 * self.a + self.b * emb2 + self.c * com_comb
        #emb = emb.unsqueeze(1)  # Adjust dimensions if necessary
        # emb = torch.stack([emb1,(com1+com2)/2,emb2], dim=1)
        emb = torch.stack([emb1,emb2], dim=1)
        #emb = torch.stack([com_comb], dim=1)
        emb, att = self.att(emb)
        # print("*********")
        # print(emb.shape)
        emb = self.MLP(emb)
        print(emb.shape)
        
        [pi, disp, mean] = self.ZINB(emb)
        #print(mean.shape)
        #print()
        return emb1, emb2, emb, pi, disp, mean
