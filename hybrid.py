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

    # 这里的 forward 必须在类里面缩进
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        res = x
        x = self.norm(x)

        # 正向
        out_fwd = self.mamba_fwd(x)

        # 反向
        x_flipped = x.flip(dims=[1])
        out_bwd = self.mamba_bwd(x_flipped).flip(dims=[1])

        return res + self.dropout(out_fwd + out_bwd)


# =========================
# 2) 定义 HeAR-TMT 混合模型
# =========================
class HeARTMTHybrid(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", mamba_layers=4):
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # 获取维度 (HeAR 是 ViT-L，所以这里通常是 1024)
        d_model = config.hidden_size

        self.base = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            add_pooling_layer=False
        )

        # B. 拆解层
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])

        # C. 插入 Mamba 层 (现在 d_model 已经定义过了)
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model) for _ in range(mamba_layers)
        ])

        self.layernorm = self.base.layernorm

    def forward(self, pixel_values):
        # 1. Embedding 层
        x = self.embeddings(pixel_values)
        # 此时 x 应该是 (Batch, 97, 1024)

        # 2. 第一段 Transformer (T)
        for blk in self.first_T:
            # 必须显式取 [0]，因为 ViTLayer 返回 (hidden_states, attention_probs)
            x = blk(x)[0]

            # === 【关键修复位置】 ===
        # 调试用：print(f"DEBUG: Shape before Mamba: {x.shape}")

        # 有些 ViT 实现会在序列第一个位置放 CLS token，
        # 如果你发现维度变成了 2 维，说明 blk(x)[0] 的实现可能触发了 pooler
        # 我们在这里确保 x 依然是 3 维
        if x.dim() == 2:
            # 如果不幸变成了 2 维，通常是因为取到了 CLS 向量
            # 我们需要检查 self.base 的配置，确保 add_pooling_layer=False
            raise ValueError(f"维度坍塌！得到了 {x.shape}。请检查是否触发了 Pooling。")

        # 3. 中间段 BiMamba (M)
        for mamba_blk in self.middle_M:
            # Mamba 层严格要求 (Batch, Seq_len, Dim)
            x = mamba_blk(x)

        # 4. 最后一段 Transformer (T)
        for blk in self.last_T:
            x = blk(x)[0]

        # 5. 输出
        x = self.layernorm(x)
        return x


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
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # 得到 TMT 混合特征
        features = self.backbone(x)

        # 聚合：在时间步维度取平均
        global_feat = features.mean(dim=1)

        return self.classifier(global_feat).squeeze(-1)


# =========================
# 4) 实例化与测试
# =========================
if __name__ == "__main__":
    print("正在初始化 HeAR-TMT Hybrid 模型...")
    # 实例化
    tmt_backbone = HeARTMTHybrid()
    model = HeARHybridClassifier(tmt_backbone).cuda()

    # 模拟输入 (Batch=2, Channel=1, H=192, W=128)
    dummy_input = torch.randn(2, 1, 192, 128).cuda()

    print("开始前向传播测试...")
    logits = model(dummy_input)
    print(f"测试成功！输出 Logits 形状: {logits.shape}")