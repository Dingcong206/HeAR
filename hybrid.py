import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# =========================
# 1) 定义双向 Mamba 块 (M 层)
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # 反向 Mamba
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(0.1)


def forward(self, pixel_values):
    # 1. Embedding 层
    x = self.embeddings(pixel_values)
    # 此时 x 的形状通常是 (Batch, 197, 1024) 或 (Batch, 97, 1024)

    # 2. 第一段 Transformer (T)
    for blk in self.first_T:
        # 关键：transformers 库的 ViTLayer 返回的是 tuple (hidden_states, weights)
        layer_outputs = blk(x)
        x = layer_outputs[0]

        # --- 修复代码开始 ---
    # 检查维度。如果 x 是 (Batch, Dim)，说明在 T 层之后被错误聚合了
    # 如果维度没问题，但 Mamba 报错，通常是因为 T 层的输出包含了 cls_token
    # 我们在这里强制检查并确保 x 是 3 维的
    if x.dim() == 2:
        # 如果意外变成了 2 维，我们需要找到原因。但在大多数 ViT 实现中，
        # x 应该是 (Batch, Seq_Len, Dim)
        raise ValueError(f"Transformer 层输出维度错误，预期3维，实际得到 {x.shape}")
    # --- 修复代码结束 ---

    # 3. 中间段 BiMamba (M)
    for mamba_blk in self.middle_M:
        x = mamba_blk(x)

    # 4. 最后一段 Transformer (T)
    for blk in self.last_T:
        layer_outputs = blk(x)
        x = layer_outputs[0]

    # 5. 输出
    x = self.layernorm(x)
    return x

# =========================
# 2) 定义 HeAR-TMT 混合模型
# =========================
class HeARTMTHybrid(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", mamba_layers=4):
        super().__init__()

        # A. 加载原始配置和模型
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # 这里的 base 是原生的 ViT-L/16
        self.base = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)

        d_model = config.hidden_size  # 应该是 1024
        num_total_layers = len(self.base.encoder.layer)  # 24层

        # B. 拆解原有 Transformer 层
        # 我们采用 T (8层) -> M (4层) -> T (8层) 的逻辑，剩余层根据需要调整
        # 这里为了演示，我们取前 8 层和后 8 层
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])

        # C. 插入自定义的 Mamba 层
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model) for _ in range(mamba_layers)
        ])

        # D. 最后的层归一化
        self.layernorm = self.base.layernorm

    def forward(self, pixel_values):
        # 1. 输入预处理 (将音频频谱图转换为 Patch Embeddings)
        # pixel_values shape: (Batch, 1, 192, 128) -> 取决于你的预处理
        x = self.embeddings(pixel_values)  # (Batch, 97, 1024)

        # 2. 第一段 Transformer (T)
        # 这里的 layer(x)[0] 是因为 transformer 返回的是 tuple
        for blk in self.first_T:
            x = blk(x)[0]

        # 3. 中间段 BiMamba (M) - 你的核心创新点
        # 这一部分捕捉时序上的选择性依赖
        for mamba_blk in self.middle_M:
            x = mamba_blk(x)

        # 4. 最后一段 Transformer (T)
        # 负责进行全局特征融合
        for blk in self.last_T:
            x = blk(x)[0]

        # 5. 输出
        x = self.layernorm(x)
        return x  # 输出 shape: (Batch, 97, 1024)


# =========================
# 3) 用于分类的完整模型 (Head)
# =========================
class HeARHybridClassifier(nn.Module):
    def __init__(self, tmt_backbone):
        super().__init__()
        self.backbone = tmt_backbone
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # 二分类
        )

    def forward(self, x):
        # 获取 97x1024 的特征
        features = self.backbone(x)

        # 聚合：取平均池化作为全局特征
        global_feat = features.mean(dim=1)

        return self.classifier(global_feat).squeeze(-1)


# =========================
# 4) 实例化与测试
# =========================
if __name__ == "__main__":
    # 实例化混合骨干网络
    tmt_backbone = HeARTMTHybrid()

    # 实例化分类器
    model = HeARHybridClassifier(tmt_backbone).cuda()

    # 模拟输入 (Batch, 1, Height, Width)
    dummy_input = torch.randn(2, 1, 192, 128).cuda()

    # 前向传播
    logits = model(dummy_input)
    print(f"输出 Logits 形状: {logits.shape}")  # 应为 [2]