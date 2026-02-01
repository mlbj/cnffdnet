"""
cnFFDNet denoising model
------------------------
Originally based on the FFDNet denoising method from the IPOL publication:
Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

Modified and restructured by Mauro Brandão (2026)
Note: This version has been heavily modified from the original source;

This program is free software: you can use, modify and/or redistribute 
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details:
<http://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import math
from spectral import conv_spectral_norm
import random

class DeinterConcatNoiselevel(torch.nn.Module):
    """
    This block takes an input batch of shape (batch_size,n_channels,n_rows,n_columns) and 
    """
    def __init__(self, sca=2):
        super().__init__()
        self.sca = sca

    def forward(self, y, noise_level):
        N, C, H, W = y.size()
        dtype = y.type()
        sca = self.sca
        sca2 = sca*sca
        Cout = sca2*C
        Hout = H//sca
        Wout = W//sca
        idxL = [[i,j] for i in range(0,sca) for j in range(0,sca)]

        # Fill the downsampled image with zeros
        if 'cuda' in dtype:
            downsampledfeatures = torch.zeros(N, Cout, Hout, Wout, 
                                              device=y.device,
                                              dtype=y.dtype)
        else:
            downsampledfeatures = torch.zeros(N, Cout, Hout, Wout) 

        if N == 1 and noise_level.shape == torch.Size([Hout,Wout]):
            noise_map = noise_level.view(1, 1, Hout, Wout)
        elif noise_level.shape == torch.Size([N,Hout,Wout]):
            noise_map = noise_level.view(N, 1, Hout, Wout)
        else:
            # Build the C x H/sca x W/sca noise map
            noise_map = noise_level.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)

        # Populate output
        for idx in range(sca2):
            downsampledfeatures[:, idx:Cout:sca2, :, :] = \
                y[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

        # concatenate de-interleaved mosaic with noise map
        return torch.cat((noise_map, downsampledfeatures), 1)

class Deinter(torch.nn.Module):
    def __init__(self, sca=2):
        super().__init__()
        self.sca = sca

    def forward(self, y, noise_level=None):
        N, C, H, W = y.size()
        dtype = y.type()
        sca = self.sca
        sca2 = sca*sca
        Cout = sca2*C
        Hout = H//sca
        Wout = W//sca
        idxL = [[i,j] for i in range(0,sca) for j in range(0,sca)]


        # Fill the downsampled image with zeros
        downsampledfeatures = torch.zeros(N, Cout, Hout, Wout,
                                          device=y.device,
                                          dtype=y.dtype)

        # Populate output
        for idx in range(sca2):
            downsampledfeatures[:, idx:Cout:sca2, :, :] = \
                y[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

        return downsampledfeatures

class UpSampleFeaturesFunction(torch.autograd.Function):
    """
    Extends PyTorch by implementing a torch.autograd.Function.
    This class implements the forward and backward methods of the last layer
    of FFDNet. It performs the inverse operation of ......... 
    takes a batch input of shape C x H/sca x W/sca and transform it into C/(sca^2) x H x W
    """
    
    # Manually set sca parameter
    sca = 2
    
    @staticmethod
    def forward(ctx, input, sca):
        #sca = UpSampleFeaturesFunction.sca
        sca = int(sca)
        ctx.sca = sca
        
        N, Cin, Hin, Win = input.size()
        
        sca2 = sca*sca
        Cout = Cin//sca2
        Hout = Hin*sca
        Wout = Win*sca
        
        result = input.new_zeros((N, Cout, Hout, Wout))
        
        idxL = [[i,j] for i in range(0, sca) for j in range(0, sca)]

        for idx in range(sca2):
            result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = \
                input[:, idx:Cin:sca2, :, :]

        return result

    @staticmethod
    def backward(ctx, grad_output):
        sca = ctx.sca
        sca2 = sca * sca
        
        N, Cg_out, Hg_out, Wg_out = grad_output.size()

        Cg_in = sca2*Cg_out
        Hg_in = Hg_out//sca
        Wg_in = Wg_out//sca
        idxL = [[i,j] for i in range(0,sca) for j in range(0,sca)]

        # Build output
        grad_input = grad_output.new_zeros((N, Cg_in, Hg_in, Wg_in))
        
        # Populate output
        for idx in range(sca2):
            grad_input[:, idx:Cg_in:sca2, :, :] = \
                grad_output[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

        return grad_input, None

# Alias functions
upsamplefeatures = UpSampleFeaturesFunction.apply

class UpSampleFeatures(torch.nn.Module):
    """
    Implements the inverse of the deinterleaving procedure as a torch module 
    """

    def __init__(self, sca=2):
        super().__init__()
        self.sca = int(sca)

    def forward(self, x):
        return  upsamplefeatures(x, self.sca)

class conicFFDNet(torch.nn.Module):
    def __init__(self, depth=15, rectifiers=None, image_channels=1, 
                 n_channels=64, bias=False, sn=True, lip=1.0, 
                 sigmas=None, comb_weights=None, sca=2, patch_shape=(66,66)):

        super().__init__()
        kernel_size = 3
        padding = 1
        self.conv_layers = []
        self.rectifiers = [nn.LeakyReLU(inplace=True) for _ in range(depth)]

        self.depth = depth
        self.n_channels = n_channels
        self.image_channels = image_channels
        self.sn = sn
        self.sca = sca
        sca2 = sca*sca

        self.power_initial_niter = 1
        self.left_singular_vectors = [None]*self.depth

        fix_noise_channel = 1
        self.deinterleaver = DeinterConcatNoiselevel(sca=sca)
        self.interleaver = UpSampleFeatures(sca=sca)
        
        rect_counter=0

        if sigmas is None:
            if lip > 0.0:
                sigmas = [pow(lip, 1.0/depth) for _ in range(depth)]
            else:
                sigmas = [0.0 for _ in range(depth)]

        # First layer
        conv = nn.Conv2d(in_channels = sca2*image_channels + fix_noise_channel*image_channels,
                         out_channels = n_channels,
                         kernel_size = kernel_size,
                         padding = padding,
                         bias = bias)
        if sn is True:
            layer = conv_spectral_norm(conv, sigma=sigmas[0], patch_shape=patch_shape)
        else:
            layer = conv
        self.conv_layers.append(layer)

        # Mid layers
        for idx in range(depth - 2):
            conv = nn.Conv2d(in_channels = n_channels,
                             out_channels = n_channels,
                             kernel_size = kernel_size,
                             padding = padding,
                             bias = bias)
            if sn is True:
                layer = conv_spectral_norm(conv, sigma=sigmas[idx+1], patch_shape=patch_shape)
            else:
                layer = conv
            self.conv_layers.append(layer)

        # Last layer
        conv = nn.Conv2d(in_channels = n_channels,
                         out_channels = sca2*image_channels,
                         kernel_size = kernel_size,
                         padding = padding,
                         bias=bias)
        if sn is True:
            layer = conv_spectral_norm(conv, sigma=sigmas[-1], patch_shape=patch_shape)
        else:
            layer = conv
        self.conv_layers.append(layer)
        
        # Comb weights 
        if comb_weights is not None:
            weights = [nn.Parameter(torch.tensor(element), requires_grad=True) for element in comb_weights]
        else:
            weights = [nn.Parameter(torch.tensor(torch.randn(1)), requires_grad=True) for _ in range(depth)]
        self.comb_weights = nn.ParameterList(weights)


        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.rectifiers = nn.Sequential(*self.rectifiers)
        
        self._initialize_weights()

    def forward(self,y,noise_level=None):
        # Concat noise level and deinterleave
        y_deinter = self.deinterleaver(y, noise_level)
 
        # Main layers
        # Top channels
        x_deinter = []
        x_deinter.append(self.conv_layers[0](y_deinter))
        x_deinter[0] = self.rectifiers[0](x_deinter[-1])

        # Mid channels (With residual connections)
        for idx in range(1, self.depth - 1):
            comb_weight = self.comb_weights[idx]**2
            ref_comb_weight = 1.0 - comb_weight

            x_deinter.append(comb_weight * self.rectifiers[idx](self.conv_layers[idx](x_deinter[idx-1])) + ref_comb_weight*x_deinter[idx-1])  

        x_deinter.append(self.rectifiers[-2](self.conv_layers[-2](x_deinter[-2])))

        # Bottom channels
        x_deinter_final = self.conv_layers[-1](x_deinter[-1])
        x_deinter_final = self.rectifiers[-1](x_deinter_final)

        # Interleave 
        x=self.interleaver(x_deinter_final)

        return x

    def _initialize_weights(self):
        for layer_idx,m in enumerate(self.conv_layers):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, a=0, mode='fan_in')
