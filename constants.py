"""
constants.py

基于 000.ipynb（dinov3.ipynb）中给出的 DINOv3 模型名称与层数映射，统一管理：
- 模型名常量
- ALL_DINOV3_MODELS 列表
- MODEL_TO_NUM_LAYERS：用于确定 ViT 的 transformer blocks 数（ConvNeXt 为 depths 之和）

用途：
- 在提取特征时，调用 model.get_intermediate_layers 需要传入层数范围 n=range(num_layers)，因此需要根据模型名确定 num_layers。
"""

# --- 模型名称常量 ---
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITLP = "dinov3_vitl16plus"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_DINOV3_CNVXT_T = "dinov3_convnext_tiny"
MODEL_DINOV3_CNVXT_S = "dinov3_convnext_small"
MODEL_DINOV3_CNVXT_B = "dinov3_convnext_base"
MODEL_DINOV3_CNVXT_L = "dinov3_convnext_large"

# --- 所有模型列表（可选）---
ALL_DINOV3_MODELS = [
    MODEL_DINOV3_VITS,
    MODEL_DINOV3_VITSP,
    MODEL_DINOV3_VITB,
    MODEL_DINOV3_VITL,
    MODEL_DINOV3_VITLP,
    MODEL_DINOV3_VITHP,
    MODEL_DINOV3_VIT7B,
    MODEL_DINOV3_CNVXT_T,
    MODEL_DINOV3_CNVXT_S,
    MODEL_DINOV3_CNVXT_B,
    MODEL_DINOV3_CNVXT_L,
]

# --- 模型到层数的映射（ConvNeXt 使用总层数）---
# ViT: depth = transformer blocks 数
# ConvNeXt: depth = sum(depths)
MODEL_TO_NUM_LAYERS = {
    # ViT models
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITLP: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,

    # ConvNeXt models (depths from official DINOv3 config)
    MODEL_DINOV3_CNVXT_T: 3 + 3 + 9 + 3,    # = 18
    MODEL_DINOV3_CNVXT_S: 3 + 3 + 27 + 3,   # = 36
    MODEL_DINOV3_CNVXT_B: 3 + 3 + 27 + 3,   # = 36
    MODEL_DINOV3_CNVXT_L: 3 + 3 + 27 + 3,   # = 36
}