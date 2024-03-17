import re
import numpy as np 

import torch 
import torch.nn as nn
from torch.nn import GELU

from nystrom_attention import NystromAttention

import pdb 

class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class NystromLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        num_landmarks=20,
        residual=True,
        dropout=0.,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_landmarks=num_landmarks,  
            pinv_iterations=6,
            residual=residual,
            dropout=dropout
        )

    def forward(self, x=None, mask=None, return_attention=False):
        # if return_attention:
        #     x, attn = self.attn(x=self.norm(x), mask=mask, return_attn=True)
        #     return x, attn
        # else:
        #     x = self.attn(x=self.norm(x), mask=mask)
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dim_head, num_landmarks, dropout):
        super(TransMIL, self).__init__()

        self.pos_layer = PPEG(dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self._fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())

        self.layer1 = NystromLayer(
            dim=hidden_dim,
            dim_head=heads,
            heads=heads,
            num_landmarks=num_landmarks,
            residual=True,
            dropout=dropout,
        )
        self.layer2 = NystromLayer(
            dim=hidden_dim,
            dim_head=heads,
            heads=heads,
            num_landmarks=num_landmarks,
            residual=True,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_attention=False):

        if len(x) == 1:
            x = x[0].unsqueeze(dim=0)
        else:
            raise NotImplementedError('TransMIL doesnt support batching')

        h = self._fc1(x) #[B, n, dim]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, n, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, n, 512]
        # if return_attention:
        #     h = self.layer2(h) #[B, n, 512]
        # else:
        #     h, attn = self.layer2(h, return_attention=True)

        #---->cls_token
        h = self.norm(h)[:,0]
        
        #---->
        # if mask is not None:
        #     h = torch.sum(self.norm(h) * ~mask.unsqueeze(dim=2), dim=1) / torch.sum(~mask, dim=1).unsqueeze(dim=1)
        # else:
        # h = self.norm(h).mean(dim=1)
 
        # if return_attention:
        #     return h, attn  

        return h 
