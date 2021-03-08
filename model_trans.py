# -*- coding: utf-8 -*-
#### source: https://github.com/hongwang600/Summarization/blob/master/attention.py
#### source: https://github.com/HazyResearch/structured-nets/blob/master/pytorch/old/misc/attention/attention.py
"""
Created on Fri Jun 12 16:47:18 2020

@author: 86139
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import math
import numpy as np
__all__ = ['make_model']

def attention(query, key, value, dropout=None):
    "计算关注度点积"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "self attention"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value):
        "self attention"
        nbatches = query.size(0)
        
        # 1) 在d_model的batch中进行线性映射
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 将attention应用在batch的映射向量中
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        
        # 3) 结果层 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
  
class PositionwiseFeedForward(nn.Module):
    "feed forward"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model) #将（d_ff,d_model）改为(d_ff,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))   
    
#标准化
class LayerNorm(nn.Module):
    "标准化"
#    原始
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        t_max = x.max()
        t_min = x.min()
        return (x - t_min) / (t_max-t_min) 
#   打算归一化

def clones(module, N):
    "N个相同层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    残差网络+标准化
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "残差网络"
        return x + self.dropout(sublayer(self.norm(x)))

class each_layer(nn.Module):
    "每层的self attention 和feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(each_layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size 
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        
#        self.BatchNorm1d = nn.BatchNorm1d(size) #归一化 1
    def forward(self, x):
        "子层"
#        mu = x.mean(-1,keepdim=True) #归一化尝试第二弹
#        std = x.std(-1,keepdim=True)
#        x = (x - mu)/std
#        x = self.BatchNorm1d(x) #归一化 2 不知道有没有用
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
#不加LSTM的

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)

        return y, attn

#### source: https://github.com/li-plus/DSNet/blob/main/src/modules/models.py
class BaseMultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024):
        super().__init__()
        self.num_head = num_head

        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)

        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k)

        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=False),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _, seq_len, num_feature = x.shape  # [1, seq_len, 1024]
        K = self.K(x)  # [1, seq_len, 1024]
        Q = self.Q(x)  # [1, seq_len, 1024]
        V = self.V(x)  # [1, seq_len, 1024]

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)  # [num_head, seq_len, d_k]
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(
            0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)

        return y, attn

class AttentionExtractor(BaseMultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out

class MRN_trans_noLSTM(nn.Module):
    "transformer attention"
    def __init__(self, layer, N,d_model):
        super(MRN_trans_noLSTM, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.fc = nn.Linear(d_model, 1)
        #self.rnn = nn.LSTM(d_model, 500, num_layers=1, bidirectional=True, batch_first=True) #试试加不加

        self.base_model = AttentionExtractor(num_head=8, num_feature=d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        "每层的输入"
#        for layer in self.layers:
#            x = layer(x)
#        return self.norm(x)

        #### Add Temporal information by skip connection source: https://github.com/li-plus/DSNet/blob/main/src/anchor_based/dsnet.py
        out = self.base_model(x)
        #x = out + x
        #x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        #x,_ = self.rnn(x)
        p = F.sigmoid(self.fc(x))
        return p

def make_model(N=6,d_model=1000, d_ff=2000, h=8, dropout=0.1):
    "构建模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    #position = PositionalEncoding(d_model, dropout)
    model = MRN_trans_noLSTM(each_layer(size=d_model,self_attn=c(attn),feed_forward=c(ff),dropout=dropout),N=N,d_model = d_model)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            
    return model