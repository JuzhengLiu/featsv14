import argparse
import os
import os.path as osp
import time
import random
import numpy as np
import torch
import torch.utils.data as tdata
import torch.nn.functional as F
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from utils.utils import update_args, Recorder, move_to, DATASET, MODEL, LOSS
import data.dataset as _data_reg
import utils.losses as _loss_reg
import models as _model_reg


HOME = osp.expanduser('/root/autodl-tmp/0-pipei-dinov3')

def load_feat(savedir: str, view: str):
    feat = torch.load(osp.join(savedir, f'{view}_feat'), map_location='cpu').to(torch.float32)
    gid = torch.load(osp.join(savedir, f'{view}_id'), map_location='cpu')
    name = torch.load(osp.join(savedir, f'{view}_name'))
    return feat, gid, name

def compute_mAP(index, good_index, junk_index):
    """
    使用标准 Average Precision 定义计算 mAP
    (Standard Definition: sum(Precision @ k) / N_gt)
    """
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    
    # 如果没有正确匹配的样本，直接返回 0
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # 1. 移除 junk_index (对结果无影响的样本)
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    # 2. 找到 good_index 在排序列表中的位置
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    # 3. 计算 CMC (Cumulative Match Characteristic)
    # 只要找到第一个正确匹配，之后的所有 rank 都视为已召回
    cmc[rows_good[0]:] = 1

    # 4. 计算 Standard AP
    # 核心修改：不再使用梯形积分，而是计算所有正确位置的 Precision 平均值
    for i in range(ngood):
        # i+1 是当前的正确样本数 (TP)
        # rows_good[i]+1 是当前检索到的总样本数 (TP + FP)
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap += precision

    ap = ap / ngood
    
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

def evaluate_model(model, opt, device):
    sat_mode = 'sat_160k' if opt.extend else 'sat'
    dro_mode = 'dro_split' if opt.train.split < 1 else 'dro'
    savedir = osp.join(HOME, opt.eval.data_path, opt.eval.feat)
    g_view = sat_mode if sat_mode == 'sat_160k' else 'sat'
    q_view = dro_mode if dro_mode == 'dro_split' else 'dro'
    gall_feat, gid, _ = load_feat(savedir, g_view)
    query_feat, qid, _ = load_feat(savedir, q_view)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, torch.nn.DataParallel):
            mod = model.module
        else:
            mod = model

        if hasattr(mod, 'shared_enc') and hasattr(mod, 'moe_layer'):
            g_enc = mod.shared_enc(gall_feat.to(device))
            q_base = mod.shared_enc(query_feat.to(device))
            # === 修复：这里改为接收 3 个返回值，忽略 aux_loss 和 load ===
            q_delta, _, _ = mod.moe_layer(q_base) 
            q_enc = q_base + q_delta
        elif hasattr(mod, 'enc1') and hasattr(mod, 'enc2'):
            res_g = mod.enc1(gall_feat.to(device))
            g_enc = res_g[0] if isinstance(res_g, tuple) else res_g
            res_q = mod.enc2(query_feat.to(device))
            q_enc = res_q[0] if isinstance(res_q, tuple) else res_q
        else:
            g_enc = gall_feat.to(device)
            q_enc = query_feat.to(device)

        g_enc = F.normalize(g_enc, dim=-1).cpu()
        q_enc = F.normalize(q_enc, dim=-1).cpu()
        
    gl = gid.cpu().numpy()
    ql = qid.cpu().numpy()
    CMC = torch.IntTensor(len(gid)).zero_()
    ap = 0.0
    for i in range(len(qid)):
        ap_tmp, CMC_tmp = eval_query(q_enc[i], ql[i], g_enc, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    AP = ap / len(qid)
    CMC = CMC.float() / len(qid)
    return AP, CMC

def train(model, dataloader, criterions, optimizer, scheduler, scaler, recorder, device, opt):
    # === [Strategy A] Dynamic Lambda Configuration ===
    start_lambda = 0.03   # 开局大权重，防止 Epoch 3 坍塌
    end_lambda = 0.01   # 结尾小权重，允许专家特化
    warmup_epochs = 20   # 前 10 个 Epoch 保持强力负载均衡
    # ===============================================
    
    for epoch in range(opt.train.epochs):
        # 注入 current_epoch 供 Loss 模块使用
        opt.current_epoch = epoch
        
        # === 计算当前 Lambda ===
        if epoch < warmup_epochs:
            current_lambda_moe = start_lambda
        else:
            current_lambda_moe = end_lambda
            
        # 每个 Epoch 打印一次当前 Lambda
        if epoch % 5 == 0:
            print(f"\n[Config] Epoch {epoch+1}: Current Lambda MoE = {current_lambda_moe:.6f}")
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recorder.reset()
        current_lr = scheduler.get_last_lr() if scheduler is not None else [opt.train.lr]
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                    desc=f'Epoch {epoch+1}/{opt.train.epochs}')
        
        for batch_idx, data in pbar:
            start_t = time.time()
            
            # 统一的前向/Loss计算逻辑
            def compute_step(data):
                data = move_to(data, device)
                data = model(data)
                loss = criterions(data, model.logit_scale, opt)
                
                if 'aux_loss' in data:
                    # 使用动态的 current_lambda_moe
                    moe_loss = data['aux_loss'] * current_lambda_moe
                    loss += moe_loss
                    recorder.update('L_moe', moe_loss.item(), opt.train.batch_size)
                    
                    # === 打印专家负载监控 ===
                    # 确保 models/autoenc_moe.py 已经把 expert_loads 放入 data
                    if batch_idx == 0 and 'expert_loads' in data:
                        loads = data['expert_loads'].detach().cpu().numpy().astype(int)
                        print(f"\n[Step {batch_idx}] Expert Loads: {loads} (Total: {loads.sum()})")
                        
                return loss

            if scaler:
                with autocast():
                    loss = compute_step(data)
                scaler.scale(loss).backward()
                if opt.train.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.train.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            else:
                loss = compute_step(data)
                loss.backward()
                if opt.train.clip_grad:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.train.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            
            recorder.update('Loss', loss.item(), opt.train.batch_size)
            recorder.update('FPS', opt.train.batch_size / (time.time() - start_t))
            
            postfix_dict = {
                'lr': f"{current_lr[0]:.6f}",
                'loss': f"{recorder.meter_dict['Loss']['meter'].avg:.4f}",
                'acc': f"{recorder.meter_dict['real_acc']['meter'].avg:.2%}",
            }
            if 'sk_eps' in recorder.meter_dict:
                postfix_dict['eps'] = f"{recorder.meter_dict['sk_eps']['meter'].avg:.3f}"
            
            pbar.set_postfix(postfix_dict)

        # 在每个 Epoch 结束时打印详细 Loss 指标
        log_msg = f"Epoch {epoch+1}: "
        loss_keys = ['L_hard', 'L_soft', 'Lrec_A', 'Lrec_B']
        for key in loss_keys:
            if key in recorder.meter_dict:
                avg_val = recorder.meter_dict[key]['meter'].avg
                log_msg += f"{key}: {avg_val:.4f} | "
        print(log_msg)
        
        if ((epoch + 1) % opt.train.save_epoch == 0) or ((epoch + 1) == opt.train.epochs):
            state = {'model': model.state_dict(), 'epoch': epoch + 1}
            save_dir = osp.join(opt.result_path, opt.cfg.split('/')[-1].split('.')[0])
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, osp.join(save_dir, f'{epoch+1}_param.t'))
        if ((epoch + 1) % opt.train.eval_epoch == 0) or ((epoch + 1) == opt.train.epochs):
            AP, CMC = evaluate_model(model, opt, device)
            top5i = 4 if len(CMC) > 4 else max(len(CMC) - 1, 0)
            top10i = 9 if len(CMC) > 9 else max(len(CMC) - 1, 0)
            top1p = len(CMC) // 100 if len(CMC) >= 100 else max(len(CMC) - 1, 0)
            print(f'Retrieval: top-1:{CMC[0]:.2%} | top-5:{CMC[top5i]:.2%} | top-10:{CMC[top10i]:.2%} | top-1%:{CMC[top1p]:.2%} | AP:{AP:.2%}')


def parse():
    parser = argparse.ArgumentParser(description='Cross-Modality Self-Supervised Training')
    parser.add_argument('cfg', type=str, help='yaml config path')
    parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--gpus', '-g', default='0', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--result_path', default='outputs', type=str)
    parser.add_argument('--extend', action='store_true')
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--feat', type=str)
    parser.add_argument('--train_feat', type=str)
    parser.add_argument('--eval_feat', type=str)
    args = parser.parse_args()
    opt = update_args(args)
    if getattr(opt, 'out_dim', None) is not None:
        opt.model['out_dim'] = opt.out_dim
    if getattr(opt, 'feat', None) is not None:
        opt.train['feat'] = opt.feat
    if getattr(opt, 'train_feat', None) is not None:
        opt.train['feat'] = opt.train_feat
    if getattr(opt, 'eval_feat', None) is not None:
        opt.eval['feat'] = opt.eval_feat
    return opt


def main():
    opt = parse()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    gpu_str = str(opt.gpus).strip()
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

    os.makedirs(opt.result_path, exist_ok=True)

    exp_name = opt.cfg.split('/')[-1].split('.')[0]

    trainset = DATASET[opt.train.dataset](**opt.train)
    trainloader = tdata.DataLoader(trainset, batch_size=opt.train.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True, pin_memory=True)

    start_epoch = 0
    model = MODEL[opt.model.name](**opt.model)
    if opt.resume:
        load_param = opt.resume
        if osp.isfile(load_param):
            checkpoint = torch.load(load_param)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)

    optimizer = model.build_opt(opt)

    if torch.cuda.device_count() > 1 and ',' in gpu_str:
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 监控列表增加 L_hard, L_soft
    recorder = Recorder({"FPS": 'd', "Loss": 'f', "L_moe": 'f', "L_hard": 'f', "L_soft": 'f', "sk_eps": 'f'})
    criterions = LOSS[opt.train.loss[0]](opt.train.loss_w[0], opt, recorder, device)

    scaler = GradScaler(init_scale=2.**10) if opt.train.mixed_precision else None

    train_steps = len(trainloader) * opt.train.epochs
    warmup_steps = len(trainloader) * opt.train.warm_epochs
    if opt.train.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_training_steps=train_steps, lr_end=opt.train.lr_end, power=1.5, num_warmup_steps=warmup_steps)
    elif opt.train.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)
    elif opt.train.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    train(model, trainloader, criterions, optimizer, scheduler, scaler, recorder, device, opt)


if __name__ == '__main__':
    main()