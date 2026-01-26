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
        # 确保输入是 (B, L, D)
        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


# =========================
# 2) 定义 HeAR-TMT 混合模型
# =========================
class HeARTMTHybrid(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", mamba_layers=4):
        super().__init__()

        # 加载配置
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = config.hidden_size  # 1024

        # 加载基础模型，禁用池化
        self.base = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True, add_pooling_layer=False
        )

        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])

        # 使用正确的 d_model 初始化 Mamba
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model) for _ in range(mamba_layers)
        ])

        self.layernorm = self.base.layernorm

    def forward(self, pixel_values):
        # pixel_values: (B, 1, 192, 128)
        x = self.embeddings(pixel_values)  # 应为 (B, 97, 1024)

        # 第一段 Transformer
        for blk in self.first_T:
            layer_outputs = blk(x)
            x = layer_outputs[0]

            # --- 维度检查：确保 x 依然是 3 维 (B, L, D) ---
        if x.dim() != 3:
            # 如果变成了 2 维，强制升维（假设此时丢失了 Batch 维是不正常的）
            # 或者抛出更详细的错误
            raise ValueError(f"维度异常！当前 x 形状为 {x.shape}，预期应为 3 维 (Batch, Seq, Dim)")

        # 中间段 BiMamba
        for mamba_blk in self.middle_M:
            x = mamba_blk(x)

        # 最后一段 Transformer
        for blk in self.last_T:
            x = blk(x)[0]

        x = self.layernorm(x)
        return x


# =========================
# 3) 分类器
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
        features = self.backbone(x)  # (B, 97, 1024)
        global_feat = features.mean(dim=1)  # (B, 1024)
        return self.classifier(global_feat).squeeze(-1)


# =========================
# 4) 运行测试
# =========================
if __name__ == "__main__":
    print("正在初始化 HeAR-TMT Hybrid 模型...")
    tmt_backbone = HeARTMTHybrid()
    model = HeARHybridClassifier(tmt_backbone).cuda()

    # 模拟输入：确保 Batch 为 2
    dummy_input = torch.randn(2, 1, 192, 128).cuda()

    print("开始前向传播测试...")
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"测试成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")