import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual, Full
from task.loss import HeatmapLoss
from models.coordconv import CoordConvTranspose

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
        
        # fc layer to latent space
        self.fcin1 = nn.ModuleList( [Full(256*4*4, 128) for i in range(nstack)] )
        
        # decoder network
        self.decoders = nn.ModuleList( [
        nn.Sequential(
            Full(128, 256*4*4),
            UnFlatten(),
            CoordConvTranspose(256, 128, kernel_size=3, stride=1, bn=True),
            CoordConvTranspose(128, 64, kernel_size=5, stride=1, bn=True),
            nn.ReLU(),
            CoordConvTranspose(64, 16, kernel_size=7, stride=1, bn=True),
        ) for i in range(nstack)] )
        
        # upsampling
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg, lowest = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            
            # latent space to decoder and upsampling as attention
            latent = self.fcin1[i](lowest)
            vae_preds = self.decoders[i](latent)
            vae_preds = self.upsampling(self.upsampling(vae_preds))
            
            preds = preds * vae_preds
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
