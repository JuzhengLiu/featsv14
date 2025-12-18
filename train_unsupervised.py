import argparse
import os
import os.path as osp
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict

from utils.utils import update_args, mkdir_if_missing
from utils.c_adamw import CAdamW
from models.unsupervised_fusion import SRDAE
from utils.unsupervised_losses import UnsupervisedLoss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_feat_layer(root, layer_idx, mode):
    path = osp.join(root, str(layer_idx))
    feat_path = osp.join(path, f'{mode}_feat')
    return torch.load(feat_path, map_location='cpu').float()

def prepare_training_data(feat_root, layers):
    print(f"==> Loading features from {feat_root}...")
    sat_feats_list = []
    dro_feats_list = []
    
    for layer in layers:
        s_f = load_feat_layer(feat_root, layer, 'sat')
        d_f = load_feat_layer(feat_root, layer, 'dro')
        sat_feats_list.append(F.normalize(s_f, dim=-1))
        dro_feats_list.append(F.normalize(d_f, dim=-1))
    
    train_X = torch.cat([torch.cat(sat_feats_list, dim=-1), 
                         torch.cat(dro_feats_list, dim=-1)], dim=0)
    print(f"==> Unsupervised Data: {train_X.shape}")
    return train_X

def train(model, dataloader, criterion, optimizer, scheduler, device, opt):
    print("==> Start SR-DAE V10 Training...")
    
    for epoch in range(opt.train.epochs):
        model.train()
        loss_avg = defaultdict(float)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_x in pbar:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            
            z, recon = model(batch_x)
            loss, loss_dict = criterion(batch_x, recon, z, model)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            for k, v in loss_dict.items():
                loss_avg[k] += v.item()
            
            # 显示关键指标，包括 Sparse Loss
            pbar.set_postfix({
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                'Rec': f"{loss_dict['recon']:.4f}", 
                'Cov': f"{loss_dict['decov']:.4f}",
                'Spa': f"{loss_dict['sparse']:.4f}"
            })
            
        msg = f"Epoch {epoch+1}: "
        for k, v in loss_avg.items():
            msg += f"{k}: {v/len(dataloader):.4f} | "
        print(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    opt = update_args(args)
    set_seed(42)


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

    
    mkdir_if_missing(opt.output_dir)
    
    train_X = prepare_training_data(opt.feat_root, opt.fusion_layers)
    if train_X.shape[1] != opt.input_dim: opt.input_dim = train_X.shape[1]
    
    dataset = TensorDataset(train_X)
    dataloader = DataLoader(dataset, batch_size=opt.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = SRDAE(input_dim=opt.input_dim, latent_dim=opt.latent_dim, drop_rate=opt.train.drop_rate).to(device)
    optimizer = CAdamW(model.parameters(), lr=opt.train.lr)
    
    train_steps = len(dataloader) * opt.train.epochs
    warmup_steps = int(len(dataloader) * opt.train.warmup_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    
    # 传入 lambda_sparse
    lambda_sparse = getattr(opt.train, 'lambda_sparse', 0.0)
    
    criterion = UnsupervisedLoss(
        lambda_recon=opt.train.lambda_recon,
        lambda_decov=opt.train.lambda_decov,
        lambda_orth=opt.train.lambda_orth,
        lambda_sparse=lambda_sparse
    ).to(device)
    
    train(model, dataloader, criterion, optimizer, scheduler, device, opt)
    
    save_path = osp.join(opt.output_dir, opt.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"==> Saved to {save_path}")

if __name__ == '__main__':
    main()