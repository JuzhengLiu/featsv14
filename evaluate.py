"""
evaluate.py

命令行脚本：
- 读取由 extract_and_save.py 保存的 dro/sat 特征文件（与 EM-CVGL test.py 格式一致），
- 计算检索指标：top-1、top-5、top-10、top-1%、AP。

说明：
- 相似度使用内积（余弦等价，因向量已 L2 归一化）。
- 计算方式参考 EM-CVGL/test.py 的 eval_query/compute_mAP。
"""

import argparse
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm


HOME = osp.expanduser('~')


def load_feat(savedir: str, view: str):
    feat = torch.load(osp.join(savedir, f'{view}_feat'), map_location='cpu').to(torch.float32)
    gid = torch.load(osp.join(savedir, f'{view}_id'), map_location='cpu')
    name = torch.load(osp.join(savedir, f'{view}_name'))
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
    # 内积作为相似度（已归一化向量）
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()

    index = np.argsort(score)[::-1]

    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)

    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc


def main():
    parser = argparse.ArgumentParser(description='DINOv3 特征检索评估')
    parser.add_argument('--save_dir', type=str, required=True, help='保存特征的目录（相对 HOME）')
    parser.add_argument('--S2D', action='store_true', help='satellite -> drone 模式')

    args = parser.parse_args()
    savedir = osp.join(HOME, args.save_dir)

    # 视角名
    gall_view = 'sat' if not args.S2D else 'dro'
    query_view = 'dro' if not args.S2D else 'sat'

    gall_feat, gid, gall_name = load_feat(savedir, gall_view)
    query_feat, qid, query_name = load_feat(savedir, query_view)

    # 展示统计信息
    gal_num = gid.shape[0]
    print(f'-----------------------')
    print(f'dim:{query_feat.shape[-1]}|#Ids\t| #Img ')
    print(f'-----------------------')
    print(f'Query\t|{torch.unique(qid).shape[0]}\t|{query_feat.shape[0]} ')
    print(f'Gallery\t|{gal_num}\t|{gal_num}')
    print(f'-----------------------')

    # 计算 CMC 与 AP（参考 test.py）
    gl = gid.cpu().numpy()
    ql = qid.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(gid)).zero_()
    ap = 0.0
    for i in tqdm(range(len(qid))):
        ap_tmp, CMC_tmp = eval_query(query_feat[i], ql[i], gall_feat, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(qid)
    CMC = CMC.float()
    CMC = CMC / len(qid)

    top1p = len(gid) // 100
    print(f'Retrieval: top-1:{CMC[0]:.2%} | top-5:{CMC[4]:.2%} | top-10:{CMC[9]:.2%} | top-1%:{CMC[top1p]:.2%} | AP:{AP:.2%}')


if __name__ == '__main__':
    main()