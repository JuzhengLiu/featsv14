import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN1
import torch.nn.functional as F
import numpy as np
from utils.utils import MODEL
from torch.optim import * 
from .moe_layer import MoE_Layer

from utils.c_adamw import CAdamW


@MODEL.register
class AutoEnc_MoE(nn.Module):
    
    def __init__(self, out_dim=1024, vec_dim=2048, num_steps=1,  
                 norm=True, normvec=True, residul=False, identify=False, query_num=701, 
                 temperature_init=14.28, bias_init=-10,
                 num_experts=4, top_k=2, 
                 noisy_gating=False,
                 # Transformer Args
                 num_chunks=20, nhead=4, dropout=0.1,
                 **opt):
        super(AutoEnc_MoE, self).__init__()
        self.num_steps = num_steps

        # === 1. 共享主干 (Shared Backbone) ===
        self.shared_enc = Seq(
            Lin(out_dim, vec_dim),
            nn.LayerNorm(vec_dim)
#            nn.GELU()  # 简单的非线性变换
        )

        # === 2. MoE 修正模块 (仅无人机, Gated-ViT Version) ===
        self.moe_layer = MoE_Layer(input_dim=vec_dim, output_dim=vec_dim, 
                                   num_experts=num_experts, k=top_k,
                                   noisy_gating=noisy_gating,
                                   num_chunks=num_chunks,
                                   nhead=nhead,
                                   dropout=dropout)
        
        # Decoder
        # Decoder
        self.dec2 = Seq(
                    Lin(vec_dim, 512),
                    BN1(512),
                    nn.GELU(),
                    Lin(512, out_dim)
                   # ,BN1(out_dim)
                )
        self.dec1 = nn.Sequential(
            nn.Linear(vec_dim, 4096), # 中间层
            BN1(4096),
            nn.GELU(),             # 在解码端引入非线性
            nn.Dropout(0.1),
            nn.Linear(4096, out_dim), # 回到原维度
            BN1(out_dim)   # 最后的归一化 (可选，视是否需要再次归一化而定)
        )        
        # self.dec2 = self.dec1
    #    self.dec2 = self.dec1 
        
        self.res = False 
        self.identify = identify
        self.query_num = query_num

        temperature = torch.nn.Parameter(torch.ones([]) * np.log(temperature_init))
        bias = torch.nn.Parameter(torch.ones([]) * bias_init)
        self.logit_scale = {"t":temperature, "b":bias}
        
        self.l2norm = lambda x: F.normalize(x, dim=-1) if norm else lambda s: s
        self.l2normvec = lambda x: F.normalize(x, dim=-1) if normvec else lambda x: x

    def run_train(self, data):
        src_x = data['x_s'] 
        tgt_x = data['x_t'] 
        
        # 处理可能的 batch 维度不一致问题
        if tgt_x.dim() == 3:
            b, n, c = tgt_x.shape
            tgt_x = tgt_x.reshape(b*n, c)[:self.query_num] 
        else:
            tgt_x = data['x_t'][:self.query_num] 
        data['x_t'] = tgt_x

        # 1. 卫星: 只走共享主干
        enc_a = self.shared_enc(src_x)
        
        # 2. 无人机: 共享主干 + MoE修正
        feat_base = self.shared_enc(tgt_x)
        # MoE 返回的是修正量 (Residual Delta)
#        feat_delta, aux_loss = self.moe_layer(feat_base)***
        feat_delta, aux_loss, expert_loads = self.moe_layer(feat_base)
        enc_b = feat_base + feat_delta 
        
        data['aux_loss'] = aux_loss
        data['expert_loads'] = expert_loads # <--- 存入 data

        # Cycle Process (训练闭环)
        fake_AA = self.dec1(enc_a) 
        fake_AB = self.dec2(enc_a) 
        fake_BA = self.dec1(enc_b) 
        fake_BB = self.dec2(enc_b) 

        # Second Pass (用于循环一致性损失)
        # AB (Sat->Dro): Sat特征伪装成Dro后，需要经过Dro的MoE路径
        feat_ab_base = self.shared_enc(fake_AB)
        feat_ab_delta, _, _ = self.moe_layer(feat_ab_base)
        enc_ab = feat_ab_base + feat_ab_delta
        fake_ABA = self.dec1(enc_ab) 

        # BA (Dro->Sat): Dro特征伪装成Sat后，只经过Sat的共享路径
        enc_ba = self.shared_enc(fake_BA)
        fake_BAB = self.dec2(enc_ba) 

        data['fake_AA'] = self.l2norm(fake_AA)
        data['fake_BB'] = self.l2norm(fake_BB)
        data['fake_ABA'] = self.l2norm(fake_ABA)
        data['fake_BAB'] = self.l2norm(fake_BAB)
        data['norm_src'] = self.l2norm(src_x)
        data['norm_tgt'] = self.l2norm(tgt_x)
        data['enc_b'] = self.l2norm(enc_b)
        data['enc_a'] = self.l2norm(enc_a)
        data['enc_ba'] = self.l2norm(enc_ba)
        
        return data

    def run_eval(self, data):
        src = data['x'] 
        
        if self.identify:
            enc_x = src
            fake_src = src 
        elif data['mode'][0] == 'sat': 
            # 卫星路径 (Clean)
            enc_x = self.shared_enc(src)
            fake_src = self.dec1(enc_x)
        else: 
            # 无人机路径 (Corrected)
            feat_base = self.shared_enc(src)
            feat_delta, _, _ = self.moe_layer(feat_base)
            enc_x = feat_base + feat_delta
            fake_src = self.dec2(enc_x)

        data['out'] = self.l2norm(enc_x)
        data['vec'] = self.l2normvec(fake_src)

        return data 

    def forward(self, data):       
        if self.training:
            data = self.run_train(data)
        else:
            data = self.run_eval(data)
        return data 

# [核心修改] 修改 build_opt 方法
# 确保 build_opt 能调用到 CAdamW
    def build_opt(self, opt):
        params = [
            {'params': self.shared_enc.parameters(), 'lr': opt.train.lr},
            {'params': self.moe_layer.parameters(), 'lr': opt.train.lr},
            {'params': self.dec1.parameters(), 'lr': opt.train.lr},
           # {'params': self.dec2.parameters(), 'lr': opt.train.lr},
            {'params': self.logit_scale['t'], 'lr': opt.train.lr},
            {'params': self.logit_scale['b'], 'lr': opt.train.lr}
        ]
        # 这里 eval(opt.train.optim) 将执行 'CAdamW'，需要 CAdamW 在当前作用域可见
        optimizer = eval(opt.train.optim)(params)
        return optimizer