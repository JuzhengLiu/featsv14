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

# --- 默认配置与路径常量 ---

# SR-DAE 模型路径 (Feature B 生成器)
DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/unsupervised_srdae_v12/srdae_model_v12.pth'

# 特征路径配置
# D2S 模式下的测试集特征路径
FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_test/dinov3_vith16plus'
# S2D 模式下的测试集特征路径
FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_test_s2d/dinov3_vith16plus'

# 融合层配置 (SR-DAE 输入层)
FUSION_LAYERS = [22, 26, 30]
INPUT_DIM_B = 3840   # 1280 * 3
LATENT_DIM_B = 1280  # SR-DAE 输出维度

# MoE 配置 (Feature A)
MOE_CONFIG_PATH = '/root/autodl-tmp/0-pipei-dinov3/configs/base_moe_shared_specific.yml'
MOE_WEIGHT_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/base_moe_shared_specific/300_param28.t'
LAYER_MAIN = 28 # Feature A 的主层


def load_feat(savedir, view):
    """
    加载指定目录下的特征与ID
    view: 'sat' 或 'dro'
    """
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")
        
    feat = torch.load(feat_path, map_location='cpu').float()
    gid = torch.load(id_path, map_location='cpu')
    return feat, gid

def compute_mAP(index, good_index, junk_index):
    """
    计算 mAP 和 CMC
    """
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    
    # 移除 junk
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    
    # 找到 good 匹配
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

def compute_mAP_standard(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # 移除 junk
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    # 找到 good 匹配
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    cmc[rows_good[0]:] = 1
    
    # --- 核心修改开始 ---
    # 使用标准 AP 定义：所有正确位置的 Precision 求和取平均
    for i in range(ngood):
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap += precision
    
    ap = ap / ngood
    # --- 核心修改结束 ---

    return ap, cmc

def eval_query(qf, ql, gf, gl):
    """
    单张 Query 对 Gallery 的评估
    """
    # 计算相似度 (已归一化，直接矩阵乘法)
    # gf 需要在 GPU 上以加速计算
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    
    # 排序
    index = np.argsort(score)[::-1]
    
    # 获取 Ground Truth
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    
    ap, cmc = compute_mAP_standard(index, good_index, junk_index)
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description='Eval SR-DAE (Unsupervised) + MoE with Optional Modes')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--srdae_path', default=DEFAULT_SRDAE_PATH, type=str, help='SR-DAE 模型路径')
    
    # [新增功能 1] 是否拼接 SR-DAE 向量
    # 默认不加参数时为 False (即执行 Fusion)，加上 --no_fusion 则为 True (不执行 Fusion)
    parser.add_argument('--no_fusion', action='store_true', help='如果不设置，默认进行 Feature A (MoE) + Feature B (SRDAE) 的拼接 (3840 dim)；如果设置此参数，仅评估 Feature A (2560 dim)。')
    
    # [新增功能 2] 评估模式切换 D2S / S2D
    parser.add_argument('--mode', default='D2S', choices=['D2S', 'S2D'], help='评估模式: D2S (Drone query, Sat gallery) 或 S2D (Sat query, Drone gallery)。默认为 D2S。')

    args = parser.parse_args()
    
    # === GPU 设置 ===
    gpu_str = str(args.gpu).strip()
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
    
    device = torch.device(device)
    print(f"Running on {device}")
    
    # === 路径选择 ===
    if args.mode == 'S2D':
        current_feat_root = FEAT_ROOT_S2D
        print(f"==> Evaluation Mode: S2D (Satellite -> Drone)")
    else:
        current_feat_root = FEAT_ROOT_D2S
        print(f"==> Evaluation Mode: D2S (Drone -> Satellite)")
        
    print(f"==> Feature Root: {current_feat_root}")
    
    # === 是否启用融合 ===
    use_fusion = not args.no_fusion
    if use_fusion:
        print("==> Strategy: Fusion Enabled (MoE 2560 + SR-DAE 1280 = 3840)")
    else:
        print("==> Strategy: Fusion Disabled (MoE 2560 only)")

    # -------------------------------------------------------------
    # 1. Feature B: SR-DAE (仅当 use_fusion=True 时执行)
    # -------------------------------------------------------------
    sat_B = None
    dro_B = None
    
    if use_fusion:
        print(f"==> Generating Feature B (SR-DAE Unsupervised) from {args.srdae_path}...")
        
        if not osp.exists(args.srdae_path):
            raise FileNotFoundError(f"Model file not found: {args.srdae_path}. Please check the path.")

        # 初始化 SR-DAE
        srdae = SRDAE(input_dim=INPUT_DIM_B, latent_dim=LATENT_DIM_B).to(device)
        
        try:
            srdae.load_state_dict(torch.load(args.srdae_path, map_location=device))
        except RuntimeError as e:
            print(f"\n[Error] Loading State Dict failed. Detail: {e}")
            return

        srdae.eval()
        
        # 加载多层特征用于 SR-DAE 输入
        sat_list, dro_list = [], []
        sat_id, dro_id = None, None
        
        print(f"  -> Loading Fusion Layers: {FUSION_LAYERS}")
        for layer in FUSION_LAYERS:
            path = osp.join(current_feat_root, str(layer))
            # 加载原始特征
            s_f, s_i = load_feat(path, 'sat')
            d_f, d_i = load_feat(path, 'dro')
            
            # Pre-Norm (拼接前归一化)
            sat_list.append(F.normalize(s_f, dim=-1))
            dro_list.append(F.normalize(d_f, dim=-1))
            
            if sat_id is None: sat_id, dro_id = s_i, d_i
                
        # 拼接多层特征
        sat_cat_b = torch.cat(sat_list, dim=-1)
        dro_cat_b = torch.cat(dro_list, dim=-1)
        
        # 批量推理由 SR-DAE 提取特征
        batch_size = 256
        def get_srdae_feat(data):
            outs = []
            for i in range(0, data.shape[0], batch_size):
                batch = data[i:i+batch_size].to(device)
                with torch.no_grad():
                    z, _ = srdae(batch) 
                    z = F.normalize(z, dim=-1) # Output Norm
                outs.append(z.cpu())
            return torch.cat(outs, dim=0)

        sat_B = get_srdae_feat(sat_cat_b)
        dro_B = get_srdae_feat(dro_cat_b)
        print(f"  -> Feature B (Sat/Dro) Shape: {sat_B.shape}")
    else:
        # 如果不融合，我们需要 ID 来进行后续评估，所以只读取 ID
        path_tmp = osp.join(current_feat_root, str(LAYER_MAIN))
        _, sat_id = load_feat(path_tmp, 'sat')
        _, dro_id = load_feat(path_tmp, 'dro')

    # -------------------------------------------------------------
    # 2. Feature A: MoE (始终执行)
    # -------------------------------------------------------------
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
    
    # 加载主层特征 (Layer 28)
    path_main = osp.join(current_feat_root, str(LAYER_MAIN))
    sat_feat_main, _ = load_feat(path_main, 'sat')
    dro_feat_main, _ = load_feat(path_main, 'dro')

    batch_size = 256
    def get_moe_feat(feat, view_mode):
        """
        view_mode: 'sat' -> Shared Encoder
                   'dro' -> Shared Encoder + MoE Layer (Correction)
        """
        outs = []
        for i in range(0, feat.shape[0], batch_size):
            batch = feat[i:i+batch_size].to(device)
            with torch.no_grad():
                base = moe.shared_enc(batch)
                if view_mode == 'dro':
                    delta, _ = moe.moe_layer(base)
                    out = base + delta
                else:
                    out = base
                out = F.normalize(out, dim=-1)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)

    # 这里的 view_mode 对应图像的真实属性：卫星图用 'sat' 逻辑，无人机图用 'dro' 逻辑
    sat_A = get_moe_feat(sat_feat_main, 'sat')
    dro_A = get_moe_feat(dro_feat_main, 'dro')
    print(f"  -> Feature A (Sat/Dro) Shape: {sat_A.shape}")

    # -------------------------------------------------------------
    # 3. Concatenation & Mode Switching
    # -------------------------------------------------------------
    print(f"==> Preparing Final Features (Mode: {args.mode})...")

    # 3.1 融合 (如有)
    if use_fusion:
        # Sat: [A, B] -> Norm
        feat_sat_final = torch.cat([sat_A, sat_B], dim=-1)
        feat_sat_final = F.normalize(feat_sat_final, dim=-1)
        
        # Dro: [A, B] -> Norm
        feat_dro_final = torch.cat([dro_A, dro_B], dim=-1)
        feat_dro_final = F.normalize(feat_dro_final, dim=-1)
    else:
        # Sat: A -> Norm
        feat_sat_final = sat_A
        feat_dro_final = dro_A

    print(f"  -> Final Feature Dim: {feat_sat_final.shape[-1]}")

    # 3.2 分配 Query 和 Gallery，并将 Gallery 移动到 GPU
    if args.mode == 'D2S':
        # D2S: Query = Drone (CPU -> GPU in loop), Gallery = Satellite (GPU)
        # 卫星数量较少 (701)，放 GPU 没问题
        gallery_feat = feat_sat_final.to(device)
        gallery_id = sat_id
        
        query_feat = feat_dro_final # 保持 CPU
        query_id = dro_id
    else:
        # S2D: Query = Satellite (CPU -> GPU in loop), Gallery = Drone (GPU)
        # 无人机数量较多 (约37k)，放 GPU 大约需要 400~600MB 显存，通常没问题
        # 如果显存不够，可以将 Gallery 保持 CPU，但矩阵乘法会慢或报错，需修改 eval_query 为分块计算
        # 这里假设显存足够，直接转 GPU 以保证计算逻辑一致
        print("  -> Moving Gallery (Drone features) to GPU...")
        gallery_feat = feat_dro_final.to(device)
        gallery_id = dro_id
        
        query_feat = feat_sat_final # 保持 CPU
        query_id = sat_id

    # -------------------------------------------------------------
    # 4. Evaluation
    # -------------------------------------------------------------
    print("==> Starting Evaluation loop...")
    
    gl = gallery_id.cpu().numpy()
    ql = query_id.cpu().numpy()
    
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    
    # 遍历 Query
    for i in tqdm(range(len(ql))):
        q_vec = query_feat[i].to(device) # 将当前 query 移入 GPU
        ap_tmp, CMC_tmp = eval_query(q_vec, ql[i], gallery_feat, gl)
        if CMC_tmp[0] != -1:
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            
    AP = ap / len(ql)
    CMC = CMC.float() / len(ql)
    
    # 计算指标
    top1 = CMC[0]
    top5 = CMC[4] if len(CMC) > 4 else CMC[-1]
    top10 = CMC[9] if len(CMC) > 9 else CMC[-1]
    
    # 打印结果
    print(f'==================================================')
    print(f'Evaluation Config:')
    print(f'  Mode:   {args.mode}')
    print(f'  Fusion: {use_fusion} (Dim: {feat_sat_final.shape[-1]})')
    print(f'Results:')
    print(f'  Retrieval: top-1:  {top1:.2%}')
    print(f'             top-5:  {top5:.2%}')
    print(f'             top-10: {top10:.2%}')
    print(f'             AP:     {AP:.2%}')
    print(f'==================================================')

if __name__ == '__main__':
    main()