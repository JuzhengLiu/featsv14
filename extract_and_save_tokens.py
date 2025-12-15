"""
extract_and_save_tokens.py
"""
import argparse
import yaml
import torch
import os
import os.path as osp
import torch.utils.data as tdata
from tqdm import tqdm

from extractor import Dinov3Extractor
# 导入 dataset 模块以触发注册
import data.dataset 
from utils.utils import DATASET 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to extract_tokens.yml')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    # [新增] 可选 GPU 参数，例如 --gpu 0
    parser.add_argument('--gpu', type=str, default=None, help='Specify GPU device id (e.g. 0 or 0,1)')
    args = parser.parse_args()

    # [新增] 如果指定了 GPU，优先设置环境变量
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # [修改] 确定运行设备：命令行参数 > 配置文件
    if args.gpu is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg['model'].get('device', 'cuda')
    
    # 获取 local_path
    local_path = cfg['model'].get('local_path', None)
    # 简单的容错
    if local_path is None and os.path.exists('./dinov3'):
        local_path = './dinov3'

    print(f"Loading DINOv3 from: {local_path if local_path else 'Remote/Cache'}")
    print(f"Running on Device: {device} (GPU: {args.gpu if args.gpu else 'Default'})")

    # 实例化 Extractor
    extractor = Dinov3Extractor(
        model_name=cfg['model']['name'],
        desc_layer=cfg['model']['desc_layer'],
        dinov3_local_path=local_path,
        return_tokens=True,
        device=device
    )

    # 确定 Dataset 类
    target_dataset_name = 'U1652_Image_D2S'
    if target_dataset_name in DATASET:
        DatasetClass = DATASET[target_dataset_name]
    else:
        # Fallback
        keys = list(DATASET.keys())
        print(f"Dataset '{target_dataset_name}' not found. Using first available: {keys[0]}")
        DatasetClass = DATASET[keys[0]]

    # 构造数据路径
    base_data_path = cfg['data']['data_path']
    if args.split == 'train':
        run_data_path = os.path.join(base_data_path, 'train')
    else:
        run_data_path = os.path.join(base_data_path, 'test')

    # 准备输出目录：output_path/{split}
    # 例如: /root/.../outputs/tokens_v13/train
    save_root = cfg['output']['path']
    save_dir = os.path.join(save_root, args.split)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Processing split: {args.split} | Data Path: {run_data_path}")
    print(f"Output Directory: {save_dir}")

    # 分别提取 sat 和 dro
    views = ['sat', 'dro']
    
    for view in views:
        print(f"--- Extracting {view} ---")
        
        # 直接透传 config 中的 transform 列表 (纯字符串列表)
        dataset_cfg = {
            'data_path': run_data_path,
            'transform': cfg['transform'], 
            'mode': view 
        }
        
        try:
            dataset = DatasetClass(**dataset_cfg)
        except Exception as e:
            print(f"Error initializing dataset for {view}: {e}")
            print(f"Check path: {run_data_path}")
            continue

        if len(dataset) == 0:
            print(f"Warning: Dataset is empty for {view}. Skipping.")
            continue

        dataloader = tdata.DataLoader(
            dataset, 
            batch_size=cfg['data']['batch_size'], 
            shuffle=False, 
            num_workers=cfg['data']['workers'],
            pin_memory=True
        )

        # 提取 Tokens
        feats, ids, names = extractor.extract_loader(dataloader)
        
        # [修改] 使用 featv13 风格的保存方式
        # 保存为: {save_dir}/{view}_feat, {save_dir}/{view}_id, {save_dir}/{view}_name
        # 且不带 .pt 后缀
        extractor.save_view(save_dir, view, feats, ids, names)
        
        # feats shape: [N, 1280, 15, 15]
        print(f"Saved {view} data. Shape: {feats.shape}")

    print("All extraction tasks completed.")

if __name__ == '__main__':
    main()