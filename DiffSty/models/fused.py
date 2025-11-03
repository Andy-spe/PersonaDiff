import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + F.gelu(self.proj(x)))

class LocalTemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y = self.conv(x.permute(0,2,1)).permute(0,2,1)
        return self.norm(x + F.gelu(y))

class GlobalTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.3)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        return self.norm(x + y)


class STM(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.4):
        super().__init__()
        self.s1 = SingleTemporalBlock(dim)
        self.l1 = LocalTemporalBlock(dim)
        self.g1 = GlobalTemporalBlock(dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        s = self.s1(x)
        l = self.l1(x)
        g = self.g1(x)
        cat = torch.cat([s, l, g], dim=-1)
        y = self.fusion(cat)
        return self.norm(self.dropout(x) + y)


class SwiGLUBlock(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)  # [B, T, hidden_dim]
        x2 = self.fc2(x)  # [B, T, hidden_dim]

        # 2) SwiGLU：SiLU(x1) ⨂ x2
        x = F.silu(x1) * x2  # [B, T, hidden_dim]
        x = self.dropout(self.proj(x))  # [B, T, dim]
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.3, max_len=100):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_len = max_len

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        hidden_dim = (dim * 4) // 3
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            SwiGLUBlock(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        )

        self.norm_out = nn.LayerNorm(dim)

        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

    def forward(self, query, key):
        B, T_q, C = query.shape
        _, T_k, _ = key.shape
        device = query.device

        q = self.norm_q(query)
        k = self.norm_k(key)

        idx_q = torch.arange(T_q, device=device)
        idx_k = torch.arange(T_k, device=device)
        rel_idx = idx_q.unsqueeze(1) - idx_k.unsqueeze(0)
        rel_idx = rel_idx + (self.max_len - 1)
        rel_idx = rel_idx.clamp(0, 2 * self.max_len - 2)

        bias = self.rel_pos_bias[:, rel_idx]  # [num_heads, T_q, T_k]

        attn_mask = bias.mean(dim=0)
        attn_mask = attn_mask.unsqueeze(0).repeat(B * self.num_heads, 1, 1)  # [B*num_heads, T_q, T_k]

        attn_out, _ = self.attn(
            q, k, k,
            attn_mask=attn_mask
        )

        attn_out = self.proj_drop(self.proj(attn_out))  # [B, T_q, C]
        x = query + attn_out  # [B, T_q, C]

        x = x + self.ffn(x)                             # [B, T_q, C]

        return self.norm_out(x)                         # [B, T_q, C]



class ContentAudioFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, style_dim=256, dropout=0.5, max_len=500):
        super().__init__()

        self.mcm_con = STM(embed_dim, num_heads, dropout=dropout)
        self.mcm_aud = STM(embed_dim, num_heads, dropout=dropout)

        self.cross_attn_content = CrossAttention(embed_dim, num_heads)
        self.cross_attn_audio = CrossAttention(embed_dim, num_heads)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fusion_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output = nn.Sequential(
            nn.Linear(embed_dim, style_dim * 2),
            nn.LayerNorm(style_dim * 2),
            nn.Dropout(dropout)
        )

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)


    def forward(self, content, audio):

        content_enhanced = self.bn1(self.mcm_con(content).permute(0, 2, 1)).permute(0, 2, 1)
        audio_enhanced = self.bn2(self.mcm_aud(audio).permute(0, 2, 1)).permute(0, 2, 1)

        cross_content = self.cross_attn_content(content_enhanced, audio_enhanced)
        cross_audio = self.cross_attn_audio(audio_enhanced, content_enhanced)

        fused = self.fusion(torch.cat([cross_content, cross_audio], dim=-1))

        x = self.fusion_block(fused)
        x = x + content

        style_final = torch.mean(x, dim=1)  # [B, embed_dim]
        output = self.output(style_final)

        return output

