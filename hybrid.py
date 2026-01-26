import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# =========================
# 1) 定义双向 Mamba 块
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 严格保持 (Batch, Seq, Dim)
        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        # 在序列维度(dim=1)翻转，确保不影响 Batch 维
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


# =========================
# 2) 定义 HeAR-TMT 混合模型
# =========================
class HeARTMTHybrid(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", mamba_layers=4):
        super().__init__()

        # 加载配置获取 d_model (1024)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = config.hidden_size

        # 加载基础模型，禁用池化层以保留 97x1024 序列
        self.base = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True, add_pooling_layer=False
        )

        # 核心：提取并保留 HeAR 的原生组件
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])  # 前 8 层 Transformer
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])  # 后 8 层 Transformer

        # 创新：插入 Mamba 层
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model) for _ in range(mamba_layers)
        ])

        self.layernorm = self.base.layernorm

    def forward(self, pixel_values):
        # 1. Embedding 层
        x = self.embeddings(pixel_values)

        # 2. 第一段 Transformer (T)
        for blk in self.first_T:
            # 必须取 [0]，确保获得 hidden_states 而不是 tuple
            x = blk(x)[0]

            # --- 强制检查并修复维度 ---
        # 如果由于某种原因 Batch 维丢了，这里进行强制补回
        if x.dim() == 2:
            x = x.unsqueeze(0)

            # 3. 中间段 BiMamba (M)
        for mamba_blk in self.middle_M:
            x = mamba_blk(x)

        # 4. 最后一段 Transformer (T)
        for blk in self.last_T:
            x = blk(x)[0]

        # 5. 最终 LayerNorm，保持与 HeAR 原始输出风格一致
        x = self.layernorm(x)
        return x  # 输出 shape: (Batch, 97, 1024)


# =========================
# 3) 分类器 (Head)
# =========================
class HeARHybridClassifier(nn.Module):
    def __init__(self, tmt_backbone):
        super().__init__()
        self.backbone = tmt_backbone
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # 获取 TMT 混合特征 (B, 97, 1024)
        features = self.backbone(x)

        # 平均池化聚合：保留 HeAR 论文中常用的特征聚合思路
        global_feat = features.mean(dim=1)

        return self.classifier(global_feat).squeeze(-1)


# =========================
# 4) 测试脚本
# =========================
if __name__ == "__main__":
    print("正在初始化 HeAR-TMT Hybrid 模型...")
    tmt_backbone = HeARTMTHybrid()
    model = HeARHybridClassifier(tmt_backbone).cuda()

    # 模拟输入 (Batch=2, Channel=1, 192, 128)
    dummy_input = torch.randn(2, 1, 192, 128).cuda()

    print("开始前向传播测试...")
    with torch.no_grad():
        logits = model(dummy_input)

    print("-" * 30)
    print(f"测试成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"Backbone 输出形状: (Batch, 97, 1024)")  # 验证是否保留了序列
    print(f"分类器输出 Logits 形状: {logits.shape}")
    print("-" * 30)