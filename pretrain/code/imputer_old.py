import torch
import torch.nn as nn
from unet import UnetSkipConnectionBlock


class UNet_st(nn.Module):
    def __init__(self, input_nc=48, output_nc=1, ngf=64, layers=5,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        mid_layers = layers - 2
        fact = 2**mid_layers

        unet_block = UnetSkipConnectionBlock(
            ngf * fact, ngf * fact, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True)

        for _ in range(mid_layers):
            half_fact = fact // 2
            unet_block = UnetSkipConnectionBlock(
                ngf * half_fact, ngf * fact, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer)
            fact = half_fact

        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)
        
class UNet_sc(nn.Module):
    def __init__(self, input_nc=40, output_nc=1, ngf=64, layers=5,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        mid_layers = layers - 2
        fact = 2**mid_layers

        unet_block = UnetSkipConnectionBlock(
            ngf * fact, ngf * fact, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True)

        for _ in range(mid_layers):
            half_fact = fact // 2
            unet_block = UnetSkipConnectionBlock(
                ngf * half_fact, ngf * fact, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer)
            fact = half_fact

        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class Imputer_st(nn.Module):
    def __init__(self,channels,ncls,img_size):
        super().__init__()
        self.channels = channels
        self.ncls = ncls
        self.img_size = img_size
        self.start_size = 128
        self.transform_r = lambda x: torch.relu(x)
        self.transform = lambda x: torch.sigmoid(x)
        self.ln = nn.Sequential(nn.Linear(self.img_size**2,self.start_size**2))
        self.conv1 = nn.Sequential(nn.Conv2d(self.channels,32,3,1,1),nn.BatchNorm2d(32),nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.ncls,8,3,1,1),nn.LeakyReLU(0.2),nn.BatchNorm2d(8),nn.Upsample(scale_factor=self.start_size, mode="bilinear"))
        
        
        # 
        self.conv3 = nn.Sequential(nn.Conv2d(2, 8, 3, 1, 1), nn.LeakyReLU(0.2), nn.BatchNorm2d(8), nn.Upsample(scale_factor=self.start_size, mode="bilinear"))
        
        self.ln2 = nn.Sequential(nn.Linear((self.start_size-1)**2,self.img_size**2))

    def forward(self, input,mask, noise,label_hot, cord):
        data = input * mask + noise * (1 - mask)
        data = self.ln(data.view(data.shape[0],-1)).view(data.shape[0],self.channels,self.start_size,self.start_size)
        data = self.conv1(data)
        label_hot = label_hot.unsqueeze(2)
        label_hot = label_hot.unsqueeze(3)
        cord = cord.unsqueeze(2)
        cord = cord.unsqueeze(3)
        label = self.conv2(label_hot)
        #cord_all = self.conv2(cord)
        cord_all = self.conv3(cord)  # coords
        net = torch.cat((data,label,cord_all),1)
        net = self.imputer_net(net)
        net = self.transform_r(net)
        net = self.ln2(net.view(net.shape[0],-1)).view(net.shape[0],self.channels,self.img_size,self.img_size)
        net = self.transform(net)
        return net

class Imputer_sc(nn.Module):
    def __init__(self,channels,ncls,img_size):
        super().__init__()
        self.channels = channels
        self.ncls = ncls
        self.img_size = img_size
        self.start_size = 128
        self.transform_r = lambda x: torch.relu(x)
        self.transform = lambda x: torch.sigmoid(x)
        self.ln = nn.Sequential(nn.Linear(self.img_size**2,self.start_size**2))
        self.conv1 = nn.Sequential(nn.Conv2d(self.channels,32,3,1,1),nn.BatchNorm2d(32),nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.ncls,8,3,1,1),nn.LeakyReLU(0.2),nn.BatchNorm2d(8),nn.Upsample(scale_factor=self.start_size, mode="bilinear"))
        
        
        # 
        self.conv3 = nn.Sequential(nn.Conv2d(2, 8, 3, 1, 1), nn.LeakyReLU(0.2), nn.BatchNorm2d(8), nn.Upsample(scale_factor=self.start_size, mode="bilinear"))
        
        self.ln2 = nn.Sequential(nn.Linear((self.start_size-1)**2,self.img_size**2))

    def forward(self, input, mask, noise,label_hot):
        data = input * mask + noise * (1 - mask)
        #data = noise #+input
        data = self.ln(data.view(data.shape[0],-1)).view(data.shape[0],self.channels,self.start_size,self.start_size)
        data = self.conv1(data)
        label_hot = label_hot.unsqueeze(2)
        label_hot = label_hot.unsqueeze(3)
        #cord = cord.unsqueeze(2)
        #cord = cord.unsqueeze(3)
        label = self.conv2(label_hot)
        #cord_all = self.conv2(cord)
        #cord_all = self.conv3(cord)  # coords
        net = torch.cat((data,label),1)
        net = self.imputer_net(net)
        net = self.transform_r(net)
        net = self.ln2(net.view(net.shape[0],-1)).view(net.shape[0],self.channels,self.img_size,self.img_size)
        net = self.transform(net)
        return net


class UNetImputer_st(Imputer_st):
    def __init__(self,channels,img_size,ncls, *args, **kwargs):
        super().__init__(channels=channels,img_size=img_size,ncls=ncls)
        self.imputer_net = UNet_st(*args, **kwargs)
        
class UNetImputer_sc(Imputer_sc):
    def __init__(self,channels,img_size,ncls, *args, **kwargs):
        super().__init__(channels=channels,img_size=img_size,ncls=ncls)
        self.imputer_net = UNet_sc(*args, **kwargs)

#device = torch.device('cuda')
#channels = 1;img_size=143;ncls=4
#imputer = UNetImputer(channels=channels,img_size=img_size,ncls=ncls)
#checkpoint = torch.load("CIDR_sim/impute_model/0349.pth")
#imputer.load_state_dict(checkpoint['imputer'])
#input = torch.FloatTensor(240,1,143,143)
#mask = input
#noise = input
#label_hot = torch.FloatTensor(240,4)
#out = imputer(input,mask,noise,label_hot)
#print(out.shape)
#out = imputer(input=input,mask=mask,noise=noise)