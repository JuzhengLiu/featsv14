"""
extract_and_save_tokens_optimized.py
内存优化版：分块保存，防止 OOM (Out Of Memory)
"""
import argparse
import yaml
import torch
import os
import os.path as osp
import torch.utils.data as tdata
import gc # 引入垃圾回收
from tqdm import tqdm

from extractor import Dinov3Extractor
import data.dataset 
from utils.utils import DATASET 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to extract_tokens.yml')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    # 新增参数：每多少个 Batch 保存一次文件
    parser.add_argument('--save_interval', type=int, default=100, help='Save every N batches to save RAM')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = cfg['model'].get('device', 'cuda')
    local_path = cfg['model'].get('local_path', None)
    if local_path is None and os.path.exists('./dinov3'):
        local_path = './dinov3'

    print(f"Loading DINOv3 from: {local_path if local_path else 'Remote/Cache'}")

    extractor = Dinov3Extractor(
        model_name=cfg['model']['name'],
        desc_layer=cfg['model']['desc_layer'],
        dinov3_local_path=local_path,
        return_tokens=True,
        device=device
    )

    target_dataset_name = 'U1652_Image_D2S'
    if target_dataset_name in DATASET:
        DatasetClass = DATASET[target_dataset_name]
    else:
        keys = list(DATASET.keys())
        print(f"Dataset '{target_dataset_name}' not found. Using first available: {keys[0]}")
        DatasetClass = DATASET[keys[0]]

    base_data_path = cfg['data']['data_path']
    if args.split == 'train':
        run_data_path = os.path.join(base_data_path, 'train')
    else:
        run_data_path = os.path.join(base_data_path, 'test')

    save_root = cfg['output']['path']
    save_dir = os.path.join(save_root, args.split)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Processing split: {args.split} | Data Path: {run_data_path}")
    print(f"Output Directory: {save_dir}")

    views = ['sat', 'dro']
    
    for view in views:
        print(f"--- Extracting {view} ---")
        
        dataset_cfg = {
            'data_path': run_data_path,
            'transform': cfg['transform'], 
            'mode': view 
        }
        
        try:
            dataset = DatasetClass(**dataset_cfg)
        except Exception as e:
            print(f"Error initializing dataset for {view}: {e}")
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

        # --- 内存优化核心循环 ---
        chunk_idx = 0
        feats_cache, ids_cache, names_cache = [], [], []
        
        # 记录分块文件列表，方便后续处理
        chunk_files = []

        total_batches = len(dataloader)
        print(f"Total Batches: {total_batches}. Saving every {args.save_interval} batches.")

        for i, batch in tqdm(enumerate(dataloader), total=total_batches, desc=f"Extracting {view}"):
            x = batch['x']
            if hasattr(x, 'ndim') and x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            y = batch['y']
            name = batch['name']
            
            # 提取
            v = extractor._extract_batch_feats(x) # [B, 1280, 15, 15]
            
            # 存入缓存
            feats_cache.append(v.cpu())
            ids_cache.append(y)
            names_cache.extend(name)

            # 达到保存间隔 或 最后一个Batch
            if (i + 1) % args.save_interval == 0 or (i + 1) == total_batches:
                # 拼接
                chunk_feats = torch.cat(feats_cache, dim=0)
                chunk_ids = torch.cat(ids_cache, dim=0)
                
                # 保存分块
                # 格式: {view}_feat_part_{idx}
                f_feat = osp.join(save_dir, f'{view}_feat_part_{chunk_idx}')
                f_id = osp.join(save_dir, f'{view}_id_part_{chunk_idx}')
                f_name = osp.join(save_dir, f'{view}_name_part_{chunk_idx}')
                
                torch.save(chunk_feats, f_feat)
                torch.save(chunk_ids, f_id)
                torch.save(names_cache, f_name)
                
                chunk_files.append(chunk_idx)
                print(f"  Saved Chunk {chunk_idx}: {chunk_feats.shape}")
                
                # 清空内存 & 强制回收
                feats_cache, ids_cache, names_cache = [], [], []
                del chunk_feats, chunk_ids
                gc.collect()
                
                chunk_idx += 1

        print(f"Saved {chunk_idx} chunks for {view}.")
        
        # 保存一个索引文件，告诉后续程序有哪些块
        index_info = {
            'num_chunks': chunk_idx,
            'feat_prefix': f'{view}_feat_part_',
            'id_prefix': f'{view}_id_part_',
            'name_prefix': f'{view}_name_part_'
        }
        torch.save(index_info, osp.join(save_dir, f'{view}_chunks_index.pt'))

    print("All extraction tasks completed.")

if __name__ == '__main__':
    main()