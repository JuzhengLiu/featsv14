import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Replaces LayerNorm for better stability and lower computational cost.
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class LayerScale(nn.Module):
    """
    LayerScale: Scale the output of the residual branch.
    Initialized to a small value (1e-4) to allow the expert to start 
    as an identity mapping, ensuring training stability.
    """
    def __init__(self, dim, init_values=1e-4, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class StandardFFN(nn.Module):
    """
    Standard FeedForward Network.
    Structure: Linear(d -> 4d) -> ReLU -> Dropout -> Linear(4d -> d)
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True), # Using ReLU as requested
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerExpert(nn.Module):
    """
    Optimized Transformer Expert.
    
    Structure:
    [Vector->Seq] -> PosEmbed 
    -> [RMSNorm -> SA -> LayerScale] + Resid
    -> [RMSNorm -> FFN(ReLU) -> LayerScale] + Resid
    -> [Seq->Vector] -> Proj(Zero-Init)
    """
    def __init__(self, input_dim, hidden_dim, num_chunks, nhead=4, dropout=0.1):
        super(TransformerExpert, self).__init__()
        self.input_dim = input_dim
        self.num_chunks = num_chunks
        
        assert input_dim % num_chunks == 0, f"Input dim {input_dim} must be divisible by num_chunks {num_chunks}"
        self.chunk_dim = input_dim // num_chunks
        
        # 1. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_chunks, self.chunk_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. Attention Block (Pre-Norm with RMSNorm & LayerScale)
        self.norm1 = RMSNorm(self.chunk_dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.chunk_dim, num_heads=nhead, 
                                          dropout=dropout, batch_first=True)
        self.ls1 = LayerScale(self.chunk_dim)
        
        # 3. FeedForward Block (Standard ReLU FFN with RMSNorm & LayerScale)
        self.norm2 = RMSNorm(self.chunk_dim)
        self.ffn = StandardFFN(self.chunk_dim, hidden_dim, dropout=dropout)
        self.ls2 = LayerScale(self.chunk_dim)
        
        # 4. Final Projection (Zero-Init for Residual Learning)
        self.proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: [B, input_dim]
        B, D = x.shape
        
        # --- Vector to Sequence ---
        x_seq = x.view(B, self.num_chunks, self.chunk_dim)
        x_seq = x_seq + self.pos_embed
        
        # --- Block 1: Self-Attention ---
        # Residual Connection: x = x + LayerScale(Attn(RMSNorm(x)))
        residual = x_seq
        x_norm = self.norm1(x_seq)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = residual + self.ls1(x_attn)
        
        # --- Block 2: FeedForward ---
        # Residual Connection: x = x + LayerScale(FFN(RMSNorm(x)))
        residual = x_seq
        x_norm = self.norm2(x_seq)
        x_ffn = self.ffn(x_norm)
        x_seq = residual + self.ls2(x_ffn)
        
        # --- Sequence to Vector ---
        x_flat = x_seq.reshape(B, D)
        
        # --- Final Projection ---
        out = self.proj(x_flat)
        return out

class MoE_Layer(nn.Module):
    """
    Mixture of Transformer Experts (RMSNorm + LayerScale)
    """
    def __init__(self, input_dim, output_dim, num_experts=4, k=2, 
                 noisy_gating=False, num_chunks=20, nhead=4, dropout=0.1):
        super(MoE_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        # Router (Gating Network)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 4),
            nn.Tanh(),
            nn.Linear(num_experts * 4, num_experts)
        )
        
        # Experts Configuration
        chunk_dim = input_dim // num_chunks
        # Standard expansion ratio is 4
        ffn_dim = int(chunk_dim * 4) 
        
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = TransformerExpert(
                input_dim=input_dim, 
                hidden_dim=ffn_dim, 
                num_chunks=num_chunks,
                nhead=nhead,
                dropout=dropout
            )
            self.experts.append(expert)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # --- Router Logic ---
        logits = self.gate(x) 
        if self.training and self.noisy_gating:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        scores = F.softmax(logits, dim=-1) 
        
        # Top-K
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Aux Loss
        if self.training:
            importance = scores.sum(dim=0)
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            load = mask.sum(dim=0)
            aux_loss = (self.num_experts * (importance * load).sum()) / (batch_size * batch_size)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        # --- Expert Execution ---
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(self.k):
            idx = topk_indices[:, i] # [B]
            weight = topk_scores[:, i].unsqueeze(1) # [B, 1]
            
            for expert_idx in range(self.num_experts):
                mask = (idx == expert_idx)
                if mask.sum() > 0:
                    selected_input = x[mask]
                    expert_out = self.experts[expert_idx](selected_input)
                    output[mask] += weight[mask] * expert_out
                    
        return output, aux_loss