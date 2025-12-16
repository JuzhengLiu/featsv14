import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml
import sys

# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())
from utils.utils import update_args
from models import AutoEnc_MoE

# --- 基础配置 ---
CONFIG_PATH = 'configs/base_moe_shared_specific.yml'
WEIGHT_PATH = '/home/dell/data/jzl/0-pipei-dinov3/outputs/base_moe_shared_specific/300_param.t'

# 默认测试集特征路径
FEAT_ROOT_DEFAULT = '/home/dell/data/jzl/0-pipei-dinov3/feats_test/dinov3_vith16plus'
# [新增] 默认训练集特征路径 (假设你提取在 feats 目录下，请根据实际情况修改)
TRAIN_FEAT_ROOT_DEFAULT = '/home/dell/data/jzl/0-pipei-dinov3/feats/dinov3_vith16plus'

# 生成 Feature A (2560) 的源层
LAYER_MAIN = 28 

# 参与 PCA 融合生成 Feature B 的层列表
# FUSION_LAYERS = [24, 27, 31]
FUSION_LAYERS = [22, 26, 30]
# FUSION_LAYERS = [28, 31] top-1:84.87% | top-5:97.51% | top-10:98.74% | top-1%:98.74% | AP:87.73%
# FUSION_LAYERS = [28, 24] #Retrieval: top-1:84.94% | top-5:97.46% | top-10:98.67% | top-1%:98.67% | AP:87.75%
# FUSION_LAYERS = [28, 22] #top-1:84.59% | top-5:97.40% | top-10:98.61% | top-1%:98.61% | AP:87.48%
#FUSION_LAYERS = [28, 23] #Retrieval: top-1:84.74% | top-5:97.44% | top-10:98.63% | top-1%:98.63% | AP:87.59%
#Retrieval: top-1:84.84% | top-5:98.10% | top-10:99.04% | top-1%:99.04% | AP:87.82%

class PCA_GPU:
    """
    使用 PyTorch 在 GPU 上实现的 PCA。
    """
    def __init__(self, n_components, whiten=True, device='cuda'):
        self.n_components = n_components
        self.whiten = whiten
        self.device = device
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # X: [N, D] tensor
        n, d = X.shape
        
        # 1. 计算均值并中心化
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_

        # 2. 计算协方差矩阵
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (n - 1)

        # 3. 特征分解
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

        # 4. 排序 (降序)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 5. 计算解释方差比
        total_variance = torch.sum(eigvals)
        if total_variance == 0:
            total_variance = 1.0
            
        # 6. 截取前 n_components
        k = min(self.n_components, d)
        self.components_ = eigvecs[:, :k] 
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = torch.sum(self.explained_variance_) / total_variance

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_transformed = torch.matmul(X_centered, self.components_)
        if self.whiten:
            scale = torch.sqrt(self.explained_variance_ + 1e-7)
            X_transformed = X_transformed / scale.unsqueeze(0)
        return X_transformed

def load_feat(savedir: str, view: str):
    """读取特征文件"""
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    name_path = osp.join(savedir, f'{view}_name')
    
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"特征文件不存在: {feat_path}")
        
    feat = torch.load(feat_path, map_location='cpu').to(torch.float32)
    gid = torch.load(id_path, map_location='cpu')
    name = torch.load(name_path) if osp.exists(name_path) else None
    return feat, gid, name

def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc

def eval_query(qf, ql, gf, gl):
    device = qf.device
    gf = gf.to(device)
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc

# [修改] 增加了 train_list 参数
def perform_pca(g_list, q_list, train_list=None, n_components=1280, whiten=True, fit_on='train', device='cuda'):
    """
    执行 PCA 降维 (GPU 版本)
    """
    print(f"  -> Preparing data for PCA on {device}...")
    
    # 1. 拼接函数
    def norm_cat(layers):
        # 先对每一层做 L2 归一化，再拼接
        normed = [F.normalize(l, dim=-1) for l in layers]
        return torch.cat(normed, dim=-1)

    X_g = norm_cat(g_list)
    X_q = norm_cat(q_list)
    
    print(f"  -> Gallery/Query Feature Dim: {X_g.shape[1]}")
    
    # 2. 准备 PCA 训练数据 (Fit Data)
    if fit_on == 'train':
        print("  -> Mode: Fit on TRAINING set")
        if train_list is None:
            raise ValueError("fit_on='train' 时必须提供 train_list 数据")
        # 训练集也需要同样的 norm_cat 处理
        X_train = norm_cat(train_list)
    elif fit_on == 'all':
        print("  -> Mode: Fit on ALL (Gallery + Query) [Transductive]")
        X_train = torch.cat([X_g, X_q], dim=0)
    else: # gallery
        print("  -> Mode: Fit on GALLERY only")
        X_train = X_g

    # 3. Fit PCA (On GPU)
    print(f"  -> Fitting PCA on {X_train.shape[0]} samples (GPU)...")
    
    # 注意显存：如果训练集非常大，这里可能需要放到 CPU 做或者分批
    # 这里的实现假设显存足够 (U1652 训练集通常没问题)
    X_train_gpu = X_train.to(device)
    
    pca = PCA_GPU(n_components=n_components, whiten=whiten, device=device)
    pca.fit(X_train_gpu)
    
    del X_train_gpu
    if fit_on == 'train':
        del X_train # 释放内存
    torch.cuda.empty_cache()
    
    print(f"  -> PCA Explained Variance Ratio (Top {pca.n_components}): {pca.explained_variance_ratio_.item():.4f}")

    # 4. Transform (On GPU)
    print("  -> Transforming Gallery and Query...")
    g_out = pca.transform(X_g.to(device))
    q_out = pca.transform(X_q.to(device))
    
    # 再次归一化输出
    return F.normalize(g_out, dim=-1), F.normalize(q_out, dim=-1)

def main():
    parser = argparse.ArgumentParser(description='PCA Fusion Evaluation')
    parser.add_argument('--gpus', default='0', type=str)
    
    # PCA 参数
    parser.add_argument('--pca_dim', default=1280, type=int, help='Target dimension for Feature B after PCA')
    parser.add_argument('--no_whiten', action='store_true', help='Disable PCA whitening')
    # [修改] fit_mode 增加 'train' 选项
    parser.add_argument('--fit_mode', default='train', choices=['all', 'gallery', 'train'], help='Data to fit PCA on')
    
    parser.add_argument('--feat_root', default=FEAT_ROOT_DEFAULT, type=str, help='测试集特征路径')
    # [新增] 训练集特征路径参数
    parser.add_argument('--train_feat_root', default=TRAIN_FEAT_ROOT_DEFAULT, type=str, help='训练集特征路径 (仅当 fit_mode=train 时使用)')

    args = parser.parse_args()
    
    # 1. 环境设置
    gpu_str = str(args.gpus).strip()
    device = 'cpu'
    if torch.cuda.is_available():
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            device = 'cuda:0'
        else:
            try:
                gpu_idx = int(gpu_str)
                device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                device = 'cuda:0'
    
    print(f"Running on {device}")
    print(f"PCA Config: Dim={args.pca_dim}, Whiten={not args.no_whiten}, Fit={args.fit_mode}")

    # 2. 构建模型 (用于生成 Feature A)
    print("Building AutoEnc_MoE model...")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['out_dim'] = 1280
    config['model']['vec_dim'] = 2560
    
    model = AutoEnc_MoE(**config['model'])
    
    if osp.isfile(WEIGHT_PATH):
        checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        raise FileNotFoundError(f"Weight not found: {WEIGHT_PATH}")
    
    model.to(device).eval()

    # 3. 生成 Feature A (2560 dim)
    print(f"Generating Feature A (2560) from Layer {LAYER_MAIN}...")
    path_main = osp.join(args.feat_root, str(LAYER_MAIN))
    g_feat_main, g_id, _ = load_feat(path_main, 'sat')
    q_feat_main, q_id, _ = load_feat(path_main, 'dro')
    
    with torch.no_grad():
        g_A = model.shared_enc(g_feat_main.to(device))
        g_A = F.normalize(g_A, dim=-1)
        
        q_base = model.shared_enc(q_feat_main.to(device))
        q_delta, _ = model.moe_layer(q_base)
        q_A = q_base + q_delta
        q_A = F.normalize(q_A, dim=-1)

    # 4. 准备数据生成 Feature B
    print(f"Loading layers {FUSION_LAYERS} for PCA Fusion...")
    
    g_layers_list = []
    q_layers_list = []
    train_layers_list = None
    
    # 加载测试集层特征
    for layer in FUSION_LAYERS:
        p_layer = osp.join(args.feat_root, str(layer))
        g_l, _, _ = load_feat(p_layer, 'sat')
        q_l, _, _ = load_feat(p_layer, 'dro')
        g_layers_list.append(g_l)
        q_layers_list.append(q_l)

    # [新增] 如果需要在训练集上 Fit，则加载训练集层特征
    if args.fit_mode == 'train':
        print(f"Loading TRAINING features from {args.train_feat_root}...")
        train_layers_list = []
        for layer in FUSION_LAYERS:
            p_layer_train = osp.join(args.train_feat_root, str(layer))
            if not osp.exists(p_layer_train):
                raise FileNotFoundError(f"训练集特征未找到: {p_layer_train}。请先对训练集运行 extract_and_save.py。")
            
            # 加载训练集的 sat 和 dro 特征
            # 这里的命名 'sat'/'dro' 取决于你提取训练集时保存的名称，通常 extract_and_save.py 也会保存为 sat_feat/dro_feat
            t_sat, _, _ = load_feat(p_layer_train, 'sat')
            t_dro, _, _ = load_feat(p_layer_train, 'dro')
            
            # 这一层的训练数据 = sat + dro
            layer_train_cat = torch.cat([t_sat, t_dro], dim=0)
            train_layers_list.append(layer_train_cat)
    
    # 5. 执行 PCA 并生成 Feature B
    with torch.no_grad():
        g_B, q_B = perform_pca(g_layers_list, q_layers_list, 
                               train_list=train_layers_list,
                               n_components=args.pca_dim, 
                               whiten=(not args.no_whiten),
                               fit_on=args.fit_mode,
                               device=device)
    
    print(f"Feature B generated. Shape: {g_B.shape}")

    # 6. 拼接 A + B
    print("Concatenating A + B ...")
    g_A = g_A.to(device)
    q_A = q_A.to(device)
    
    g_final = torch.cat([g_A, g_B], dim=-1)
    g_final = F.normalize(g_final, dim=-1) 
    
    q_final = torch.cat([q_A, q_B], dim=-1)
    q_final = F.normalize(q_final, dim=-1)

    print(f"Final Feature Shape: {g_final.shape}")

    # 7. 评估
    print("Starting Evaluation...")
    gl = g_id.cpu().numpy()
    ql = q_id.cpu().numpy()
    
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    
    for i in tqdm(range(len(ql))):
        ap_tmp, CMC_tmp = eval_query(q_final[i], ql[i], g_final, gl)
        if CMC_tmp[0] != -1:
            CMC = CMC + CMC_tmp
            ap += ap_tmp

    AP = ap / len(ql)
    CMC = CMC.float() / len(ql)

    top1 = CMC[0]
    top5 = CMC[4] if len(CMC) > 4 else CMC[-1]
    top10 = CMC[9] if len(CMC) > 9 else CMC[-1]
    top1p = CMC[len(CMC) // 100]

    print(f'==================================================')
    print(f'GPU PCA Fusion Results (Layers: {FUSION_LAYERS} -> PCA {args.pca_dim} | Fit on {args.fit_mode}):')
    print(f'Retrieval: top-1:{top1:.2%} | top-5:{top5:.2%} | top-10:{top10:.2%} | top-1%:{top1p:.2%} | AP:{AP:.2%}')
    print(f'==================================================')

if __name__ == '__main__':
    main()