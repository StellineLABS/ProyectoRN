#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importar dependencias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from torch.cuda.amp import autocast
from functools import partial
import os
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.layers import use_fused_attn
from timm.models.layers import to_2tuple,trunc_normal_
import numpy as np
import wget


# In[3]:


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In[4]:


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# In[5]:


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


# In[6]:


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, 
                 norm_layer=None, bias=True,drop=0.,use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# In[7]:


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_norm=qk_norm,attn_drop=attn_drop,proj_drop=proj_drop, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# In[8]:


class PatchEmbedding(nn.Module):
    '''
    Clase para obtener el patch embedding: Dividimos la imagen en 'paches' y los proyectamos.
    '''
    def __init__(self, img_size = 224, patch_size = 16, input_channels = 3, embedding_n = 768):
        super().__init__()

        self.img_size = (img_size, img_size)#QT o_tuple
        self.patch_size = (patch_size, patch_size)
        self.n_patch = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.proj = nn.Conv2d(input_channels, embedding_n, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# In[9]:


class ASTModel(nn.Module):
    """
    Modelo de audio espectograma 
    :param class_n: numero de clases. 
    :param div_f: division del patch en dim frecuencia.
    :param div_t: division del patch en dim tiempo. 
    :param input_f: bins de frecuencia en entrada.
    :param input_t: frames de tiempo en entrada.
    """
    def __init__(self, class_n = 527, div_f = 10, div_t = 10, input_f = 128, input_t = 1024):

        super(ASTModel, self).__init__()
        
        # para el input embedding 
        self.patch_embedding = PatchEmbedding() 
        self.embedding_n = self.patch_embedding.n_patch 
        
        # perceptron 
        self.mlp = nn.Sequential(nn.LayerNorm(self.embedding_n), 
                                          nn.Linear(self.embedding_n, class_n))
        # Para el embedding
        f_dim, t_dim = self.get_shape(div_f, div_t, input_f, input_t)
        self.patch_embedding.n_patch = f_dim * t_dim
        self.patch_embedding.proj  = torch.nn.Conv2d(1, self.embedding_n, kernel_size=(16, 16), stride=(div_f, div_t))
        # inicializamos de manera aleatoria los pesos correspondientes al positional embedding 
        self.p_e = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patch + 1, self.embedding_n))
        trunc_normal_(self.p_e, std=0.02)
        
        self.pos_drop = nn.Dropout(p=0)
        
        # Para calcular el input embedding (por la modificación para transformadores de imágenes)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_n))
        
        # Parametros elegidos a base de prueba y error para la parte de atención
        self.blocks = nn.Sequential(*[ Block(dim = self.embedding_n, num_heads = 7, mlp_ratio = 4, qkv_bias = True,
                                             norm_layer = partial(nn.LayerNorm, eps=1e-6))
                                      for i in range(12)])
        #capa de normalización
        self.norm = nn.LayerNorm(self.embedding_n)
        
    def get_shape(self, div_f, div_t, input_f=128, input_t=1024):
        '''
        Obtenemos el valor más óptimo para representar las dimensiones de frecuencia y tiempo
        '''
        test_input = torch.randn(1, 1, input_f, input_t)
        test_proj = nn.Conv2d(1, self.embedding_n + 2, kernel_size=(16, 16), stride=(div_f, div_t))
        test_out = test_proj(test_input)
        return (test_out.shape[2], test_out.shape[3])

    @autocast()
    def forward(self, x):
        """
        :param x: tensor que representa el espectograma de un audio 
        :return: el sample que se cree que se encuentra en el audio representado por x
        """
        x = x.unsqueeze(1) 
        x = x.transpose(2, 3)
        
        # obtenemos el input embedding
        B = x.shape[0]
        x = self.patch_embedding(x) 
        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        
        # agregamos codificación posicional
        x = x + self.p_e
        
        x = self.pos_drop(x) 
        
        # Pasamos nuestro vector por todas las cabezas 
        for b in self.blocks:
            x = b(x)
            
        # Normalizamos la salida
        x = self.norm(x)
        x = (x[:,0] + x[:,1]) / 2
        
        # Perceptron multicapa
        x = self.mlp(x)
        
        return x


# In[ ]:




