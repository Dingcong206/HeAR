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
        x = self.embeddings(pixel_values)  # (B, 97, 1024)

        for blk in self.first_T:
            x = blk(x)[0]

        # 确保进入 Mamba 前是 (B, L, D)
        if x.dim() == 2: x = x.unsqueeze(0)

        for mamba_blk in self.middle_M:
            x = mamba_blk(x)

        for blk in self.last_T:
            x = blk(x)[0]

        x = self.layernorm(x)
        return x  # 确保这里不做任何 transpose

# =========================
# 3) 用于分类的完整模型 (必须在调用前定义)
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
        # 1. 获取特征 (B, 97, 1024)
        features = self.backbone(x)

        # --- 健壮性修复：如果 Batch 维丢失，强制补回 ---
        if features.dim() == 2:  # 变成了 (97, 1024)
            features = features.unsqueeze(0)  # 补回为 (1, 97, 1024)

        # 2. 检查维度顺序
        # 如果末尾是 97，说明 1024 在中间，需要转置
        if features.shape[-1] == 97:
            features = features.transpose(1, 2)

        # 3. 聚合：在时间序列维度(97)取平均，保留特征维度(1024)
        # 关键：dim=1 必须对应序列长度那一维
        global_feat = features.mean(dim=1)  # 得到 (B, 1024)

        # 4. 如果 mean 之后变成了 (1024,)，补回 Batch 维
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

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