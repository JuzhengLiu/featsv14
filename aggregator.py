"""
aggregator.py

功能：
- 针对 DINOv3 ViT 的最后一层特征（形状一般为 [C, H, W]），提供多种全局聚合方式：
  - Avg：全局平均池化
  - Max：全局最大池化
  - GeM：广义均值池化（Generalized Mean Pooling）
  - VLAD：向量的局部聚合描述（需聚类中心，推荐复用 AnyLoc.py 的 VLAD 类与缓存）

说明：
- 与 AnyLoc.py 中最后一层的描述符聚合思想一致，聚合后会做 L2 归一化，得到最终的全局向量用于检索。
- VLAD 聚合需要事先训练或缓存聚类中心（centroids），本实现支持复用 AnyLoc.VLAD（若工程内可用）。
"""

from typing import Callable, Optional
import torch
import torch.nn.functional as F


def _l2n(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 归一化辅助函数"""
    return F.normalize(v, dim=dim)


def avg_pool(feat: torch.Tensor) -> torch.Tensor:
    """
    平均池化聚合
    参数：
    - feat: [C, H, W]
    返回：
    - vec: [C]（L2 归一化）
    """
    vec = feat.mean(dim=(1, 2))
    return _l2n(vec, dim=0)


def max_pool(feat: torch.Tensor) -> torch.Tensor:
    """
    最大池化聚合
    参数：
    - feat: [C, H, W]
    返回：
    - vec: [C]（L2 归一化）
    """
    vec, _ = feat.max(dim=1)
    vec, _ = vec.max(dim=1)
    return _l2n(vec, dim=0)


def gem_pool(feat: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """GeM 聚合（Generalized Mean Pooling）
    vec[c] = ( mean( (feat[c,:,:]) ** p ) ) ** (1/p)
    最后做 L2 归一化
    """
    x = torch.clamp(feat, min=eps)
    x = x.pow(p).mean(dim=(1, 2)).pow(1.0 / p)
    return _l2n(x, dim=0)


class VLADPooler:
    """
    VLAD 聚合器封装。
    依赖：
    - models.AnyLoc.VLAD 类（若工程内可用），并且需要已缓存的聚类中心（cache_dir/c_centers.pt）。

    用法：
    vlad_pooler = VLADPooler(num_clusters=8, cache_dir='~/.cache/vlad')
    vec = vlad_pooler(feat)  # feat: [C, H, W]

    注意：
    - 若未找到 VLAD 类或缓存不可用，建议改用 GeM 聚合。
    """
    def __init__(self, num_clusters: int = 8, cache_dir: Optional[str] = None):
        self.num_clusters = num_clusters
        self.cache_dir = cache_dir
        self._vlad = None
        self._ready = False

        # try:
        #     from models.AnyLoc import VLAD as ANYLOC_VLAD  # 复用工程内实现
        #     self._vlad = ANYLOC_VLAD(num_clusters=num_clusters, cache_dir=cache_dir)
        #     # 仅在存在缓存的聚类中心时标记 ready
        #     self._ready = self._vlad.can_use_cache_vlad()
        # except Exception:
        #     # 无法导入 AnyLoc.VLAD，则暂不启用 VLAD
        #     self._ready = False
        self._vlad = None

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        if not self._ready or self._vlad is None:
            raise RuntimeError("VLADPooler 未就绪：请确保已安装 AnyLoc.VLAD 且提供 cache_dir/c_centers.pt。")
        # feat: [C, H, W] -> [N, D]
        C, H, W = feat.shape
        desc = feat.permute(1, 2, 0).reshape(H * W, C)  # [HW, C]
        vec = self._vlad.generate(desc)  # [num_clusters * C]
        return _l2n(vec, dim=0)


def get_aggregator(name: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    根据名称返回聚合函数/对象。
    - name: 'Avg' | 'Max' | 'GeM' | 'VLAD'
    - kwargs: 额外参数，例如 GeM 的 p，VLAD 的 num_clusters/cache_dir 等
    返回：
    - 可调用对象：输入 [C, H, W]，输出 [D]
    """
    name = str(name).lower()
    if name == 'avg':
        return avg_pool
    elif name == 'max':
        return max_pool
    elif name == 'gem':
        p = kwargs.get('p', 3.0)
        return lambda feat: gem_pool(feat, p=p)
    elif name == 'vlad':
        num_c = kwargs.get('num_c', 8)
        cache_dir = kwargs.get('cache_dir', None)
        pooler = VLADPooler(num_clusters=num_c, cache_dir=cache_dir)
        return pooler
    else:
        raise ValueError(f"未知聚合器: {name}")