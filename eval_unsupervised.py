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

# 引入新模型 (SRDAE V11 / V9 等兼容模型)
from models.unsupervised_fusion import SRDAE
from models.autoenc_moe import AutoEnc_MoE 

# --- 默认配置 ---
# 默认指向 V11 模型 (请根据实际情况修改)

# DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/unsupervised_srdae_v14/srdae_model_v14.pth'
# FEAT_ROOT_TEST = '/root/autodl-tmp/0-pipei-dinov3/feats_test/dinov3_vith16plus'
# FUSION_LAYERS = [21, 22, 23, 24, 25, 26, 27, 29, 30, 31]
# INPUT_DIM_B = 12800
# LATENT_DIM_B = 1280

DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/unsupervised_srdae_v12/srdae_model_v12.pth'
FEAT_ROOT_TEST = '/root/autodl-tmp/0-pipei-dinov3/feats_test/dinov3_vith16plus'
FUSION_LAYERS = [22, 26, 30]
INPUT_DIM_B = 3840
LATENT_DIM_B = 1280


# MoE 配置 (Feature A)
MOE_CONFIG_PATH = '/root/autodl-tmp/0-pipei-dinov3/configs/base_moe_shared_specific.yml'
MOE_WEIGHT_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/base_moe_shared_specific/300_param28.t'
LAYER_MAIN = 28

def load_feat(savedir, view):
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")
        
    feat = torch.load(feat_path, map_location='cpu').float()
    gid = torch.load(id_path, map_location='cpu')
    return feat, gid

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
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description='Eval SR-DAE (Unsupervised) + MoE')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--srdae_path', default=DEFAULT_SRDAE_PATH, type=str, help='SR-DAE 模型路径')
    args = parser.parse_args()
    
    # === [FIXED] 参考 train_pseudo.py 的 GPU 设置逻辑 ===
    gpu_str = str(args.gpu).strip()
    device = 'cpu'
    if torch.cuda.is_available():
        # 如果包含逗号(多卡)或者为空，则设置环境变量并使用默认 cuda:0
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            device = 'cuda:0'
        else:
            # 单卡模式，直接指定 cuda:X
            try:
                gpu_idx = int(gpu_str)
                device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                device = 'cuda:0'
    
    device = torch.device(device)
    print(f"Running on {device}")
    
    # 1. Feature B: SR-DAE
    print(f"==> Generating Feature B (SR-DAE Unsupervised) from {args.srdae_path}...")
    
    if not osp.exists(args.srdae_path):
        raise FileNotFoundError(f"Model file not found: {args.srdae_path}. Please check the path.")

    # 初始化模型
    srdae = SRDAE(input_dim=INPUT_DIM_B, latent_dim=LATENT_DIM_B).to(device)
    
    # 加载权重
    try:
        srdae.load_state_dict(torch.load(args.srdae_path, map_location=device))
    except RuntimeError as e:
        print(f"\n[Error] Loading State Dict failed. Detail: {e}")
        return

    srdae.eval()
    
    g_list, q_list = [], []
    g_id, q_id = None, None
    
    print(f"  -> Loading Fusion Layers: {FUSION_LAYERS}")
    for layer in FUSION_LAYERS:
        path = osp.join(FEAT_ROOT_TEST, str(layer))
        g_f, g_i = load_feat(path, 'sat')
        q_f, q_i = load_feat(path, 'dro')
        # Pre-Norm
        g_list.append(F.normalize(g_f, dim=-1))
        q_list.append(F.normalize(q_f, dim=-1))
        if g_id is None: g_id, q_id = g_i, q_i
            
    g_cat_b = torch.cat(g_list, dim=-1)
    q_cat_b = torch.cat(q_list, dim=-1)
    
    # Inference Feature B
    batch_size = 256
    def get_srdae_feat(data):
        outs = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size].to(device)
            with torch.no_grad():
                z, _ = srdae(batch) 
                z = F.normalize(z, dim=-1)
            outs.append(z.cpu())
        return torch.cat(outs, dim=0)

    g_B = get_srdae_feat(g_cat_b)
    q_B = get_srdae_feat(q_cat_b)
    print(f"  -> Feature B Shape: {g_B.shape}")

    # 2. Feature A: MoE (Existing)
    print("==> Generating Feature A (MoE)...")
    with open(MOE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['out_dim'] = 1280
    config['model']['vec_dim'] = 2560
    
    moe = AutoEnc_MoE(**config['model'])
    
    if osp.exists(MOE_WEIGHT_PATH):
        checkpoint = torch.load(MOE_WEIGHT_PATH, map_location='cpu')
        moe.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        raise FileNotFoundError(f"MoE weights not found at {MOE_WEIGHT_PATH}")
        
    moe.to(device).eval()
    
    path_main = osp.join(FEAT_ROOT_TEST, str(LAYER_MAIN))
    g_feat_main, _ = load_feat(path_main, 'sat')
    q_feat_main, _ = load_feat(path_main, 'dro')

    def get_moe_feat(feat, mode):
        outs = []
        for i in range(0, feat.shape[0], batch_size):
            batch = feat[i:i+batch_size].to(device)
            with torch.no_grad():
                base = moe.shared_enc(batch)
                if mode == 'dro':
                    delta, _ = moe.moe_layer(base)
                    out = base + delta
                else:
                    out = base
                out = F.normalize(out, dim=-1)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)

    g_A = get_moe_feat(g_feat_main, 'sat')
    q_A = get_moe_feat(q_feat_main, 'dro')

    # 3. Concat & Eval
    print("==> Concatenating & Evaluating...")
    # Concat [A, B] -> Normalize
    g_final = F.normalize(torch.cat([g_A, g_B], dim=-1), dim=-1).to(device)
    q_final = F.normalize(torch.cat([q_A, q_B], dim=-1), dim=-1)
    
    gl = g_id.cpu().numpy()
    ql = q_id.cpu().numpy()
    
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    
    for i in tqdm(range(len(ql))):
        q_vec = q_final[i].to(device)
        ap_tmp, CMC_tmp = eval_query(q_vec, ql[i], g_final, gl)
        if CMC_tmp[0] != -1:
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            
    AP = ap / len(ql)
    CMC = CMC.float() / len(ql)
    
    top1 = CMC[0]
    top5 = CMC[4] if len(CMC) > 4 else CMC[-1]
    
    print(f'==================================================')
    print(f'SR-DAE (Unsupervised) + MoE Results:')
    print(f'Retrieval: top-1:{top1:.2%} | top-5:{top5:.2%} | AP:{AP:.2%}')
    print(f'==================================================')

if __name__ == '__main__':
    main()