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
        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


# =========================
# 2) 定义 HeAR-TMT 混合模型 (Backbone)
# =========================
class HeARTMTHybrid(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", mamba_layers=4):
        super().__init__()

        # A. 加载配置并获取隐藏层维度 (1024)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = config.hidden_size

        # B. 加载基础模型，禁用池化层
        self.base = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True, add_pooling_layer=False
        )

        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])

        # C. 插入 Mamba 层
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model) for _ in range(mamba_layers)
        ])

        self.layernorm = self.base.layernorm

    def forward(self, pixel_values):
        # 输入: (B, 1, 192, 128) -> 输出: (B, 97, 1024)
        x = self.embeddings(pixel_values)

        # 第一阶段: Transformer
        for blk in self.first_T:
            x = blk(x)[0]

        # 维度容错处理 (针对 Batch=1 的情况)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # 第二阶段: BiMamba
        for mamba_blk in self.middle_M:
            x = mamba_blk(x)

        # 第三阶段: Transformer
        for blk in self.last_T:
            x = blk(x)[0]

        x = self.layernorm(x)
        return x


# =========================
# 3) 用于分类的完整模型 (必须在调用前定义)
# =========================
class HeARHybridClassifier(nn.Module):
    def __init__(self, tmt_backbone):
        super().__init__()
        self.backbone = tmt_backbone
        # 维度对齐：1024 是 HeAR 的输出维度
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # 1. 经过 TMT Backbone 得到 (B, 97, 1024)
        features = self.backbone(x)

        # 2. 在时间步维度 (dim=1) 取平均，得到 (B, 1024)
        global_feat = features.mean(dim=1)

        # 3. 分类输出
        return self.classifier(global_feat).squeeze(-1)


# =========================
# 4) 实例化与测试
# =========================
if __name__ == "__main__":
    print("正在初始化 HeAR-TMT Hybrid 模型...")
    # 按照顺序：先定义类，再实例化
    tmt_backbone = HeARTMTHybrid()
    model = HeARHybridClassifier(tmt_backbone).cuda()

    # 模拟输入 (Batch=2)
    dummy_input = torch.randn(2, 1, 192, 128).cuda()

    print("开始前向传播测试...")
    with torch.no_grad():
        logits = model(dummy_input)

    print("-" * 30)
    print("测试成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")  # 应为 [2]
    print("-" * 30)