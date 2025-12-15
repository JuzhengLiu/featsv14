"""
dinov3_feat 模块

目的：
- 基于 DINOv3 模型，完成图像特征的提取、聚合与保存；
- 读取预先保存的特征向量，计算相似度并输出排序指标（top-1/5/10/1%/AP），与 EM-CVGL 仓库 test.py 的评估流程保持一致；

设计要点：
- 严格沿用 EM-CVGL 的数据结构与变换（utils/transform.py 与 data/dataset.py 的组织方式）；
- 以 000.ipynb（dinov3.ipynb）中的 DINOv3 加载与中间层特征提取方式为主，实现 ViT 模型的中间层特征获取（get_intermediate_layers）；
- 在最后一层特征基础上提供多种全局聚合策略（Avg/Max/GeM/VLAD），与 AnyLoc.py 中的思路一致；
- 将提取出的特征按 EM-CVGL 的保存格式存盘：sat_feat/sat_id/sat_name 与 dro_feat/dro_id/dro_name；

文件结构（不超过 6 个 py 文件）：
- constants.py：定义 DINOv3 模型常量与层数映射（参考 000.ipynb）
- aggregator.py：聚合器实现（Avg/Max/GeM/VLAD）
- extractor.py：DINOv3 特征提取器类，负责加载模型、提取中间层特征并聚合
- extract_and_save.py：命令行脚本，按照 EM-CVGL 的数据与 transform，提取并保存 dro/sat 特征
- evaluate.py：命令行脚本，载入已保存特征并计算评估指标，输出 top-1/5/10/1%/AP

注意事项：
- 尽量与 EM-CVGL 的数据结构保持兼容，这样可以复用其 DataLoader、transform、与 test/eval 习惯；
- VLAD 聚合需要聚类中心（centroids），可复用 AnyLoc 中的 VLAD 类与已有缓存；若不可用则推荐使用 GeM 作为默认聚合策略；
"""

from .constants import ALL_DINOV3_MODELS, MODEL_TO_NUM_LAYERS
from .aggregator import get_aggregator
from .extractor import Dinov3Extractor