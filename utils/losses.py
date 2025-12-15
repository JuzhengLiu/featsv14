import torch
from utils.utils import LOSS
from torch import nn
import torch.nn.functional as F


def filter_InfoNCE(sim_mat, sim_mat2, logit_scale, loss_fn, label1, label2):
    # 标准 InfoNCE，保持数值稳定
    logits_per_image1 = logit_scale * sim_mat
    logits_per_image2 = logit_scale * sim_mat2
    loss = (loss_fn(logits_per_image1, label1) + loss_fn(logits_per_image2, label2)) / 2
    return loss


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """
    Sinkhorn algorithm in log-domain.
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


@LOSS.register
class CycleAELoss(nn.Module):
    def __init__(self, weight, args, recorder, device):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=args.train.label_smoothing)
        self.recon_fn = torch.nn.L1Loss()
        self.cycle_fn = torch.nn.L1Loss()
        self.pseudo_fn = None
        self.fea_cyc_fn = torch.nn.MSELoss()
        # KL 散度损失，用于软蒸馏
        self.kl_div_fn = nn.KLDivLoss(reduction='batchmean')
        
        self.device = device
        self.w = weight
        self.recorder = recorder
        self.dro_num = args.train.dro_num
        self.lambda_A = args.train.lambda_A
        self.lambda_B = args.train.lambda_B
        self.idt_w = args.train.idt_w
        self.feat_w = args.train.feat_w
        self.thr = args.train.pseudo_thr
        self.mutual_match = args.train.mutual_match
        self.keep_neg = args.train.keep_neg
        
        # 蒸馏权重
        self.distill_w = getattr(args.train, 'distill_w', 0.5)

        # === Sinkhorn Config (Dynamic) ===
        self.sk_eps_start = getattr(args.train, 'sinkhorn_eps_start', 0.2)
        self.sk_eps_end = getattr(args.train, 'sinkhorn_eps_end', 0.05)
        self.sk_iters = getattr(args.train, 'sinkhorn_iters', 3)
        self.total_epochs = args.train.epochs

    def get_current_epsilon(self, current_epoch):
        """计算当前 Sinkhorn 正则化系数 (线性衰减)"""
        if self.total_epochs == 0: return self.sk_eps_start
        progress = min(1.0, current_epoch / self.total_epochs)
        return self.sk_eps_start + (self.sk_eps_end - self.sk_eps_start) * progress

    def forward(self, data, logit_scale, args=None):
        fake_AA = data['fake_AA']
        fake_BB = data['fake_BB']
        enc_b = data['enc_b']
        enc_a = data['enc_a']
        real_A = data['x_s']
        real_B = data['x_t'].squeeze()
        A_id = data['y_s']
        B_id = data['y_t']

        # 1. 计算局部相似度矩阵 (Local Similarity)
        sim_mat = torch.einsum('md, nd-> mn', enc_a, enc_b)
        m, n = sim_mat.shape
        sim_mat_multi = sim_mat.reshape(m, n // self.dro_num, self.dro_num)
        sim_mat_mean = sim_mat_multi.mean(-1) # [m, n_unique]
        
        # 2. 计算 Sinkhorn 全局规划矩阵 (作为软目标)
        C = 1.0 - sim_mat_mean
        device = sim_mat_mean.device
        mu = torch.ones(m, device=device) / m
        nu = torch.ones(n // self.dro_num, device=device) / (n // self.dro_num)
        
        current_epoch = getattr(args, 'current_epoch', 0)
        epsilon = self.get_current_epsilon(current_epoch)
        
        with torch.no_grad():
             P_log = log_sinkhorn_iterations(-C / epsilon, torch.log(mu), torch.log(nu), self.sk_iters)
             P = torch.exp(P_log)
             # P 是双随机矩阵(近似)，行和为 1/M。
             # 为了作为分类概率目标，我们需要将其行/列归一化为 1。
             
             # Target for A->B (Row-wise probability)
             target_AB = P / P.sum(dim=1, keepdim=True)
             
             # Target for B->A (Col-wise probability)
             target_BA = (P / P.sum(dim=0, keepdim=True)).T

        # 3. GLSD Track 1: Hard Mutual Match InfoNCE (保持原有最佳逻辑)
        # 获取基于 Sinkhorn 和 Sim 的混合打分，用于 Hard Selection
        # 这里还是沿用之前的 confidence 辅助判断，或者直接用 Sim + P max 索引
        # 为了最稳定，我们恢复您验证过的逻辑：P 和 Sim 双向最大值校验
        
        # 获取 Sinkhorn 的双向最大索引
        _, idx_P_row = P.max(dim=1)
        _, idx_P_col = P.max(dim=0)
        
        # 检查互匹配: i 是 j 的最佳，j 也是 i 的最佳
        sk_mutual_mask = (idx_P_col[idx_P_row] == torch.arange(m, device=device))
        
        # 相似度阈值过滤
        sim_val = sim_mat_mean[torch.arange(m, device=device), idx_P_row]
        hard_mask = sk_mutual_mask & (sim_val > self.thr)
        
        # 计算 Hard Loss
        if hard_mask.sum() > 0:
            valid_A_idx = torch.nonzero(hard_mask).squeeze()
            if valid_A_idx.ndim == 0: valid_A_idx = valid_A_idx.unsqueeze(0)
            valid_B_idx = idx_P_row[hard_mask]
            
            logit_scale_exp = logit_scale["t"].exp()
            
            # A->B
            l_hard_a = self.loss_function(logit_scale_exp * sim_mat_mean[valid_A_idx], valid_B_idx)
            # B->A
            l_hard_b = self.loss_function(logit_scale_exp * sim_mat_mean.T[valid_B_idx], valid_A_idx)
            
            loss_hard = (l_hard_a + l_hard_b) / 2
        else:
            loss_hard = torch.tensor(0.0, device=device, requires_grad=True)

        # 4. GLSD Track 2: Soft Sinkhorn Distillation (全量样本蒸馏)
        # 强制 Sim 矩阵的分布去逼近 Sinkhorn P 矩阵的分布
        # Scale sim_mat by logit_scale before softmax to match InfoNCE behavior
        logit_scale_exp = logit_scale["t"].exp()
        
        # KL(Target || Input_Log_Probs)
        # A->B Distillation
        log_probs_AB = F.log_softmax(logit_scale_exp * sim_mat_mean, dim=1)
        loss_soft_AB = self.kl_div_fn(log_probs_AB, target_AB)
        
        # B->A Distillation
        log_probs_BA = F.log_softmax(logit_scale_exp * sim_mat_mean.T, dim=1)
        loss_soft_BA = self.kl_div_fn(log_probs_BA, target_BA)
        
        loss_soft = (loss_soft_AB + loss_soft_BA) / 2

        # 5. 重构损失
        loss_recon_A = self.idt_w * self.recon_fn(fake_AA, real_A)
        loss_recon_B = self.idt_w * self.recon_fn(fake_BB, real_B)

        # 统计真实准确率 (监控用)
        gt_mask = A_id.unsqueeze(1) == B_id[:n].unsqueeze(0)
        _, gt_idx = gt_mask.max(-1)
        real_acc = (idx_P_row == gt_idx.to(idx_P_row)).float().mean()

        self.recorder.update('Lrec_A', loss_recon_A.item(), args.train.batch_size, type='f')
        self.recorder.update('Lrec_B', loss_recon_B.item(), args.train.batch_size, type='f')
        self.recorder.update('L_hard', loss_hard.item(), args.train.batch_size, type='f')
        self.recorder.update('L_soft', loss_soft.item(), args.train.batch_size, type='f')
        self.recorder.update('real_acc', real_acc.item(), args.train.batch_size, type='%')
        self.recorder.update('sk_eps', epsilon, args.train.batch_size, type='f')

        # 总损失 = Hard Loss + Weight * Soft Loss + Recon
        final_loss = loss_hard + self.distill_w * loss_soft + loss_recon_A + loss_recon_B
        return final_loss