"""
extract_and_save.py

命令行脚本：
- 使用 EM-CVGL 的数据集与变换（U1652_Image_D2S + utils/transform.py），
- 以 000.ipynb 的 DINOv3 提取方式为主，提取 dro 与 sat 的特征，
- 在最后一层进行指定聚合（Avg/Max/GeM/VLAD），
- 并将特征按 EM-CVGL 的格式保存（sat_feat/sat_id/sat_name, dro_feat/dro_id/dro_name）。

基本用法：
python -m dinov3_feat.extract_and_save \
    configs/base_anyloc_D2S.yml \
    --model_name dinov3_vits16 \
    --dinov3_local_path "C:/path/to/dinov3-main" \
    --agg GeM \
    --save_dir "feat_dinov3"

说明：
- save_dir 为相对 HOME 的路径，最终保存到：~/<save_dir>/... 文件。
"""

import argparse
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from utils.utils import update_args, Logger
from data.dataset import *  # 直接复用 EM-CVGL 的数据集定义

from constants import MODEL_TO_NUM_LAYERS
from aggregator import get_aggregator
from extractor import Dinov3Extractor


HOME = osp.expanduser('~')


def build_dataloaders(opt, sat_mode: str = 'sat', S2D: bool = False):
    """构建查询与图库两个 DataLoader，与 test.py 保持一致"""
    queryset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
    gallset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)

    if S2D:
        gallset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
        queryset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)

    query_loader = DataLoader(queryset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)
    gall_loader = DataLoader(gallset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)
    return query_loader, gall_loader


def main():
    parser = argparse.ArgumentParser(description='DINOv3 特征提取与保存（对齐 EM-CVGL）')
    parser.add_argument('cfg', type=str, help='yaml 配置路径，例如 configs/base_anyloc_D2S.yml')
    parser.add_argument('--gpu', '-g', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model_name', default=None, type=str, help='DINOv3 模型名称（若不提供，将尝试从 YAML 的 model.dino_model 读取）')
    parser.add_argument('--dinov3_local_path', default='/root/autodl-tmp/dinov3-main',type=str, help='DINOv3 本地 torch.hub 仓库路径')
    parser.add_argument('--weights', type=str, help='DINOv3 权重文件路径（可选）')
    parser.add_argument('--agg', default='GeM', type=str, choices=['Avg', 'Max', 'GeM', 'VLAD'], help='最后一层聚合方式（不提供则从 YAML 的 model.aggre_type 读取，默认 GeM）')
    parser.add_argument('--num_c', default=None, type=int, help='VLAD 的聚类中心个数（可从 YAML 的 model.num_c 读取）')
    parser.add_argument('--vlad_cache_dir', type=str, help='VLAD 缓存目录（包含 c_centers.pt，可从 YAML 的 model.vlad_cache_dir 读取）')
    parser.add_argument('--S2D', action='store_true', help='satellite -> drone 交换')
    parser.add_argument('--extend', action='store_true', help='使用扩展 160k 卫星图库')
    #parser.add_argument('--save_dir', type=str, default='feat_dinov3', help='相对 HOME 的保存目录名')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/0-pipei-dinov3/feats', help='相对 HOME 的保存目录名')
    parser.add_argument('--desc_layer', type=int, default=None, help='中间层编号（不提供则从 YAML 的 model.desc_layer 读取，默认最后一层）')
    parser.add_argument('--desc_facet', type=str, default=None, choices=['token', 'query', 'key', 'value'], help='特征面（不提供则从 YAML 的 model.desc_facet 读取，默认 token）')
    parser.add_argument('--use_cls', action='store_true', help='是否包含 CLS token（仅在 facet 非 token 时生效；也可在 YAML 的 model.use_cls 指定）')

    args = parser.parse_args()
    opt = update_args(args)
    gpu_str = str(opt.gpu).strip()
    run_device = 'cpu'
    if torch.cuda.is_available():
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            run_device = 'cuda:0'
        else:
            try:
                gpu_idx = int(gpu_str)
                run_device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                run_device = 'cuda:0'

    log_path = osp.join('outputs', opt.cfg.split('/')[-1].split('.')[0])
    os.makedirs(log_path, exist_ok=True)
    # 记录日志（与 EM-CVGL 风格一致）
    try:
        import sys, time
        sys.stdout = Logger(osp.join(log_path, f'DINOv3_Extract_{opt.eval.dataset}_{time.asctime()}.log'))
    except Exception:
        pass

    print(f"==========\nArgs:{opt}\n==========")

    # 从 YAML/CLI 读取模型与提取配置（CLI 优先，其次 YAML，最后默认）
    def pick(val_cli, val_yaml, default=None):
        return val_cli if val_cli not in [None, ''] else (val_yaml if val_yaml is not None else default)

    # 读取 model_name（优先 CLI；其次 YAML 的 model.dino_model；默认 dinov3_vits16）
    yaml_dino_model = getattr(opt.model, 'dino_model', None) if hasattr(opt, 'model') else None
    model_name = pick(args.model_name, yaml_dino_model, 'dinov3_vits16')
    if yaml_dino_model and isinstance(yaml_dino_model, str) and not str(yaml_dino_model).startswith('dinov3_'):
        # YAML 若仍为 dinov2_xxx，则忽略 YAML 值，使用 CLI 或默认
        if args.model_name is None:
            print(f"[警告] YAML model.dino_model='{yaml_dino_model}' 非 DINOv3 模型，已改用默认/CLI: {model_name}")

    # 读取聚合方式（优先 CLI；其次 YAML 的 model.aggre_type；默认 GeM）
    yaml_aggre_type = getattr(opt.model, 'aggre_type', None) if hasattr(opt, 'model') else None
    agg_name = pick(args.agg, yaml_aggre_type, 'GeM')

    # VLAD 参数（优先 CLI；其次 YAML 的 model.num_c 与 model.vlad_cache_dir）
    yaml_num_c = getattr(opt.model, 'num_c', None) if hasattr(opt, 'model') else None
    yaml_vlad_cache = getattr(opt.model, 'vlad_cache_dir', None) if hasattr(opt, 'model') else None
    num_c = pick(args.num_c, yaml_num_c, 8)
    vlad_cache_dir = pick(args.vlad_cache_dir, yaml_vlad_cache, None)

    # 层与 facet（优先 CLI；其次 YAML 的 model.desc_layer 与 model.desc_facet）
    yaml_desc_layer = getattr(opt.model, 'desc_layer', None) if hasattr(opt, 'model') else None
    yaml_desc_facet = getattr(opt.model, 'desc_facet', None) if hasattr(opt, 'model') else None
    desc_layer = pick(args.desc_layer, yaml_desc_layer, None)
    desc_facet = pick(args.desc_facet, yaml_desc_facet, 'token')

    # use_cls（CLI 布尔默认 False；若 YAML 指定则覆盖）
    yaml_use_cls = getattr(opt.model, 'use_cls', None) if hasattr(opt, 'model') else None
    use_cls = bool(args.use_cls or (yaml_use_cls is True))

    print(f"[配置] model_name={model_name}, agg={agg_name}, num_c={num_c}, vlad_cache_dir={vlad_cache_dir}, desc_layer={desc_layer}, desc_facet={desc_facet}, use_cls={use_cls}")

    # 构建聚合器
    agg_kwargs = {}
    if str(agg_name).lower() == 'vlad':
        agg_kwargs = dict(num_c=num_c, cache_dir=vlad_cache_dir)
    aggregator = get_aggregator(agg_name, **agg_kwargs)

    # 构建 DataLoader
    # 对图片数据集（U1652_Image_*）不支持 'sat_160k' 模式，这里做兼容处理
    sat_mode = 'sat'
    if args.extend:
        ds_name = getattr(opt.eval, 'dataset', '') if hasattr(opt, 'eval') else ''
        if str(ds_name) == 'Feat_Single':
            sat_mode = 'sat_160k'
        else:
            print("[提示] 当前评估数据集不支持扩展 160k 卫星图库（仅 Feat_Single 支持），已回退为常规 'sat' 模式。")

    query_loader, gall_loader = build_dataloaders(opt, sat_mode=sat_mode, S2D=args.S2D)

    # 构建提取器
    extractor = Dinov3Extractor(
        model_name=model_name,
        dinov3_local_path=args.dinov3_local_path,
        weights=args.weights,
        device=run_device,
        aggregator=aggregator,
        desc_layer=desc_layer,
        desc_facet=desc_facet,
        use_cls=use_cls,
    )

    # 提取并保存
    print('==> 提取 gallery 特征...')
    gall_feat, gid, gall_name = extractor.extract_loader(gall_loader)
    print('==> 提取 query 特征...')
    query_feat, qid, query_name = extractor.extract_loader(query_loader)

    # 保存目录（HOME 下）
    #保存在save_dir再加上模型名称和模型层数
    print(HOME)
    print(args.save_dir)
    print(model_name)
    print(str(desc_layer))
    savedir = osp.join(HOME, args.save_dir, model_name, str(desc_layer))
    os.makedirs(savedir, exist_ok=True)

    # 视角名
    gall_view = 'sat' if not args.S2D else 'dro'
    query_view = 'dro' if not args.S2D else 'sat'

    extractor.save_view(savedir, gall_view, gall_feat, gid, gall_name)
    extractor.save_view(savedir, query_view, query_feat, qid, query_name)

    print('==> 提取与保存完成')


if __name__ == '__main__':
    main()
