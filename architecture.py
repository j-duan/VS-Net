import torch
import torch.nn as nn
from data import transforms as T

class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = T.complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), 
                               sensitivity[...,0], sensitivity[...,1])
     
        k = torch.fft(x, 2, normalized=True)
              
        v = self.noise_lvl
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        x = torch.ifft(out, 2, normalized=True)
       
        Sx = T.complex_multiply(x[...,0], x[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)     
          
        SS = T.complex_multiply(sensitivity[...,0], 
                                sensitivity[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)
        
        return Sx, SS

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x



class cnn_layer(nn.Module):
    
    def __init__(self):
        super(cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  3, padding=1, bias=True)
        )     
        
    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    
class network(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer()) 
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
        print(self.conv_blocks)
        print(self.dc_blocks)
        print(self.wa_blocks)
 
    def forward(self, x, k, m, c):
                
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
        return x    
    
    