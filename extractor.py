"""
extractor.py

Dinov3Extractor：
- 负责加载 DINOv3 模型（优先按 000.ipynb 的方式从本地 torch.hub 路径加载），
- 提取指定数据加载器（DataLoader）中的图像特征（中间层，取最后一层），
- 使用指定的聚合器（Avg/Max/GeM/VLAD）在最后一层进行全局聚合，得到检索向量，
- 并按 EM-CVGL 的保存格式将 dro/sat 的特征、id、文件名写入磁盘。

严格对齐点：
- 数据转换（transform）沿用 utils/transform.py 的定义与 configs 中 eval.transform 的配置；
- 特征保存命名沿用 test.py：sat_feat/sat_id/sat_name 与 dro_feat/dro_id/dro_name；
- 路径前缀沿用 HOME（~）+ 相对路径的组织方式，保持与 data/dataset.py 一致的可读性与兼容性。
"""

from typing import List, Tuple, Optional
import os
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from constants import MODEL_TO_NUM_LAYERS


def _mkdir_if_missing(path: str):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


class Dinov3Extractor:
    """
    DINOv3 特征提取器：
    - 支持从本地 torch.hub 仓库加载模型（source='local'），也可回退到远程（不推荐，受网路影响）；
    - 使用 ViT 的 get_intermediate_layers 获取中间层特征（reshape=True, norm=True），选择最后一层；
    - 对最后一层的 [B, C, H, W] 特征进行聚合，得到 [B, D] 的检索向量；
    - 将 dro/sat 的特征与 id、name 打包保存。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        dinov3_local_path: Optional[str] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        aggregator=None,
        desc_layer: Optional[int] = None,
        desc_facet: str = 'token',
        use_cls: bool = False,
    ) -> None:
        """
        参数：
        - model_name：DINOv3 模型名称，参考 constants.py
        - dinov3_local_path：本地 DINOv3 仓库路径，用于 torch.hub.load(source='local')
        - weights：可选权重文件路径（若模型构造支持）
        - device：'cuda' 或 'cpu'，默认自动探测
        - aggregator：聚合器（可调用对象）：输入 [C,H,W] 输出 [D]；可使用 aggregator.get_aggregator()
        """
        self.model_name = model_name
        self.local_path = dinov3_local_path
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregator = aggregator
        self.desc_layer = desc_layer
        self.desc_facet = str(desc_facet).lower()  # 'token' | 'query' | 'key' | 'value'
        self.use_cls = use_cls

        self.model = self._load_model()
        self.num_layers = MODEL_TO_NUM_LAYERS.get(self.model_name, None)
        if self.num_layers is None:
            raise ValueError(f"不支持的 DINOv3 模型名：{self.model_name}")
        # 估计 patch_size（ViT 模型）
        self.patch_size = 16
        try:
            # 对 ViT，通常为 Conv2d kernel/stride 为 patch_size
            self.patch_size = int(getattr(self.model.patch_embed.proj, 'kernel_size')[0])
        except Exception:
            pass

        self.model = self.model.eval().to(self.device)
        # Hook 存放
        self._hook_out = None
        self._fh_handle = None

    def _load_model(self):
        """加载 DINOv3 模型，优先本地路径。"""
        if self.local_path is not None:
            # 某些 hub 实现不支持 weights 参数，增加容错
            try:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True
                )
        else:
            # 远程加载（可能受网络影响，且官方 hub 名称可能变动）
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True
                )
        return model

    @torch.inference_mode()
    def _extract_batch_feats(self, batch_x: torch.Tensor) -> torch.Tensor:
        """
        从一批图像中提取最后一层的特征并聚合。
        输入：
        - batch_x: [B, 3, H, W]
        输出：
        - vecs: [B, D]
        """
        batch_x = batch_x.to(self.device)
        # 选择层索引：默认最后一层
        layer_idx = self.desc_layer
        if layer_idx is None or layer_idx < 0 or layer_idx >= self.num_layers:
            layer_idx = self.num_layers - 1

        if self.desc_facet == 'token':
            # ViT：仅获取指定层（默认最后一层）特征，reshape=True => [B, C, H', W']；norm=True => 对 token 做归一
            feats: List[torch.Tensor] = self.model.get_intermediate_layers(
                batch_x, n=[layer_idx], reshape=True, norm=True
            )
            feat_l: torch.Tensor = feats[0]  # [B, C, H', W']
        else:
            # 通过 forward hook 从指定层的 attn.qkv 提取 query/key/value 三种 facet
            feat_l: torch.Tensor = self._extract_facet_features(batch_x, layer_idx)

        # 针对每张图聚合到全局向量
        out_list = []
        for i in range(feat_l.shape[0]):
            f_i = feat_l[i].detach()  # [C, H', W']
            # 与 000.ipynb 一致：按通道做一次 L2 归一（可选）
            f_i = F.normalize(f_i, p=2, dim=0)
            vec_i = self.aggregator(f_i) if self.aggregator is not None else f_i.mean(dim=(1, 2))
            vec_i = F.normalize(vec_i, dim=0)  # 最终向量再做 L2
            out_list.append(vec_i)
        vecs = torch.stack(out_list, dim=0)  # [B, D]
        return vecs

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            # 保存 qkv 的输出：形状通常为 [B, N, 3*C]
            self._hook_out = output
        return _forward_hook

    @torch.inference_mode()
    def _extract_facet_features(self, batch_x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        通过 attn.qkv 的 forward hook 提取指定层的 facet（query/key/value）描述符，并 reshape 成 [B, C, H', W']。
        注意：此方法适用于 ViT 模型；ConvNeXt 不支持 facet 选择。
        """
        # 注册 hook
        try:
            attn_qkv = self.model.blocks[layer_idx].attn.qkv
        except Exception:
            raise RuntimeError("当前模型不支持 facet（query/key/value）提取，请使用 desc_facet='token' 或更换为 ViT 模型。")

        # 注册 forward hook
        self._hook_out = None
        self._fh_handle = attn_qkv.register_forward_hook(self._generate_forward_hook())

        # 触发一次前向（注意：直接调用 model(batch_x) 即可，hook 会捕获中间层输出）
        _ = self.model(batch_x)

        # 取出 hook 的输出
        qkv = self._hook_out  # [B, N, 3*C]
        # 移除 hook
        try:
            if self._fh_handle is not None:
                self._fh_handle.remove()
        finally:
            self._fh_handle = None

        if qkv is None:
            raise RuntimeError("未捕获到 qkv 输出，请确认模型结构或层索引是否正确。")

        # 解析 facet
        B, N, threeC = qkv.shape
        C = threeC // 3
        if self.desc_facet == 'query':
            res = qkv[:, :, :C]
        elif self.desc_facet == 'key':
            res = qkv[:, :, C:2 * C]
        elif self.desc_facet == 'value':
            res = qkv[:, :, 2 * C:]
        else:
            raise ValueError(f"未知的 desc_facet: {self.desc_facet}")

        # 是否包含 CLS token
        if self.use_cls:
            tokens = res  # [B, N, C]
        else:
            tokens = res[:, 1:, :]  # [B, N-1, C]

        # 计算网格尺寸 H'、W'（使用 patch_size 与输入图像大小）
        # 由于 EM-CVGL 的 eval.transform 产生固定尺寸（如 252x252），ViT 的 patch=16 => H'=W'=15
        H, W = batch_x.shape[-2], batch_x.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if Hp * Wp != tokens.shape[1]:
            # 兜底：根据 token 数量估计网格（假设为正方形）
            import math
            grid = int(math.sqrt(tokens.shape[1]))
            Hp, Wp = grid, tokens.shape[1] // grid

        # reshape 成 [B, C, H', W']
        feat = tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        # 归一化（与 000.ipynb 保持一致）
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def extract_loader(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        对给定的 DataLoader 提取并聚合特征
        返回：
        - all_feats: [N, D]
        - all_ids:   [N]
        - all_names: List[str]
        """
        feats, ids, names = [], [], []
        for batch in tqdm(dataloader, desc='DINOv3 提取特征'):
            x = batch['x']  # [B, 3, H, W] 或 [B, 1, 3, H, W]（S2D 数据集会多一维）
            # 兼容 U1652_Image_S2D 中的额外维度
            if hasattr(x, 'ndim') and x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            y = batch['y']  # [B]
            name = batch['name']  # List[str]
            v = self._extract_batch_feats(x)  # [B, D]
            feats.append(v.cpu())
            ids.append(y)
            names.extend(name)

        all_feats = torch.cat(feats, dim=0).to(torch.float32)
        all_ids = torch.cat(ids, dim=0)
        return all_feats, all_ids, names

    @staticmethod
    def save_view(savedir: str, view: str, feats: torch.Tensor, ids: torch.Tensor, names: List[str]):
        """
        按 EM-CVGL 的格式保存某一视角（sat/dro）的特征与元信息。
        - savedir: 目录
        - view: 'sat' 或 'dro'（也支持 'sat_160k' 与其他兼容名）
        - feats: [N, D]
        - ids: [N]
        - names: List[str]
        """
        _mkdir_if_missing(savedir)
        torch.save(feats, osp.join(savedir, f'{view}_feat'))
        torch.save(ids, osp.join(savedir, f'{view}_id'))
        torch.save(names, osp.join(savedir, f'{view}_name'))
        print(f'[保存完成] {view} 视角特征保存在: {savedir}')