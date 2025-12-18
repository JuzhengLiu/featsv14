import torch
import torch.nn as nn
import torch.nn.functional as F

class SRDAE(nn.Module):
    """
    SR-DAE V11: Aggressive Sparse Whitening AE
    架构: Linear -> BN(affine=True) -> Linear
    改动: 移除 Dropout，完全依赖 L1 Sparsity 进行特征选择
    """
    def __init__(self, input_dim=3840, latent_dim=1280, drop_rate=0.0):
        super(SRDAE, self).__init__()
        
        # 1. 旋转 (Encoder)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        
        # 2. 筛选与白化 (BN)
        self.bn_whiten = nn.BatchNorm1d(latent_dim, affine=True)
        
        # 3. 重建 (Decoder)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.ones_(self.bn_whiten.weight) # Gamma=1
        nn.init.zeros_(self.bn_whiten.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        # 移除 Dropout，让输入更纯净，便于 Gamma 精确判断谁是噪声
        
        # Encode
        z_raw = self.encoder(x)
        
        # Whiten & Select (Gamma 将在这里发挥作用)
        z = self.bn_whiten(z_raw)
        
        # Decode
        recon = self.decoder(z)
        
        return z, recon