import torch
import torch.nn as nn
import torch.nn.functional as F

class UnsupervisedLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_decov=1.0, lambda_orth=1.0, lambda_sparse=0.0):
        super(UnsupervisedLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_decov = lambda_decov
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
        self.mse = nn.MSELoss()

    def _compute_orth_loss(self, w, device):
        # w shape: [Out, In] -> 我们希望行向量正交 (或者列向量，对于 Linear 来说是行)
        # Linear(in, out): weight is [out, in]
        # 我们希望输入基向量正交 -> w.t() @ w = I ? 不，Linear 是 x @ W.t()
        # 我们希望学到的投影矩阵列向量正交: || W.t() @ W - I ||
        
        # PyTorch Linear weight is [out_features, in_features]
        # We want the 'in_features' dimensions to be projected orthogonally?
        # No, we want the basis vectors (rows of W) to be orthogonal.
        # Gram matrix of rows: W @ W.t()
        
        # 对于 Encoder (3840 -> 1280): weight [1280, 3840]
        # 我们希望这 1280 个基向量互相正交。
        w_norm = F.normalize(w, dim=1) # Normalize rows
        gram = torch.mm(w_norm, w_norm.t()) # [1280, 1280]
        eye = torch.eye(gram.shape[0], device=device)
        return F.mse_loss(gram, eye)

    def forward(self, inputs, recon, features, model):
        batch_size = inputs.size(0)

        # 1. Recon
        loss_recon = self.mse(recon, inputs)

        # 2. Decov (Batch-wise correlation)
        z = features - features.mean(dim=0, keepdim=True)
        cov_z = torch.mm(z.t(), z) / (batch_size - 1 + 1e-6)
        loss_decov = F.mse_loss(cov_z, torch.eye(cov_z.shape[0], device=inputs.device))

        # 3. Orth (Dual: Encoder & Decoder)
        loss_orth = torch.tensor(0.0, device=inputs.device)
        
        # Encoder Orthogonality
        if hasattr(model, 'encoder'):
            loss_orth += self._compute_orth_loss(model.encoder.weight, inputs.device)
            
        # Decoder Orthogonality (New!)
        # Decoder (1280 -> 3840): weight [3840, 1280]
        # 我们希望这 3840 个输出方向是由 1280 个正交基组合而成
        # 这里约束 Decoder 的列向量正交可能更合理 (tied weights logic)，或者行向量正交
        # 为了对称性，我们约束 Decoder 的输入维度方向 (columns) 正交，即 [3840, 1280].T @ [3840, 1280]
        if hasattr(model, 'decoder'):
            # Decoder weight: [3840, 1280]
            # Transpose to [1280, 3840] to compute Gram matrix of size [1280, 1280]
            loss_orth += self._compute_orth_loss(model.decoder.weight.t(), inputs.device)

        # 4. Sparsity
        loss_sparse = torch.tensor(0.0, device=inputs.device)
        if hasattr(model, 'bn_whiten') and model.bn_whiten.affine:
            loss_sparse = torch.mean(torch.abs(model.bn_whiten.weight))

        total_loss = self.lambda_recon * loss_recon + \
                     self.lambda_decov * loss_decov + \
                     self.lambda_orth * loss_orth + \
                     self.lambda_sparse * loss_sparse
        
        loss_dict = {
            "total": total_loss,
            "recon": loss_recon,
            "decov": loss_decov,
            "orth": loss_orth,
            "sparse": loss_sparse
        }
        
        return total_loss, loss_dict