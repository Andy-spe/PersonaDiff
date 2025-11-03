import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Rotary(nn.Module):
    """
    Rotary positional embeddings applying a rotation to Q/K.
    """
    def __init__(self, dim: int = 256, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Last dim {dim} != {self.dim}")
        device = x.device
        self._update_cos_sin_cache(seq_len, device)
        cos = self.cos_cached.to(x.dtype)
        sin = self.sin_cached.to(x.dtype)
        x = x.view(bsz, seq_len, dim // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        cos = cos.view(1, seq_len, dim // 2, 2)[..., 0]
        sin = sin.view(1, seq_len, dim // 2, 2)[..., 0]
        x_rot_even = x1 * cos - x2 * sin
        x_rot_odd  = x1 * sin + x2 * cos
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).view(bsz, seq_len, dim)
        return x_rot

class RotaryAttention(nn.Module):
    """
    Multi-head self-attention with Rotary embeddings and relative position bias.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, max_rel_dist=128, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rotary = Rotary(dim=self.head_dim)
        self.max_rel_dist = max_rel_dist
        self.rel_bias = nn.Parameter(torch.zeros((2 * max_rel_dist + 1, num_heads)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        B, T, C = x.shape
        context = x if context is None else context

        # fused QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # apply rotary
        q_flat = q.permute(0,2,1,3).reshape(B*self.num_heads, T, self.head_dim)
        k_flat = k.permute(0,2,1,3).reshape(B*self.num_heads, T, self.head_dim)
        q_rot = self.rotary(q_flat).view(B, self.num_heads, T, self.head_dim)
        k_rot = self.rotary(k_flat).view(B, self.num_heads, T, self.head_dim)

        v = v.permute(0,2,1,3)
        attn = (q_rot @ k_rot.transpose(-2, -1)) * self.scale

        # relative bias
        idx = torch.arange(T, device=x.device)
        rel = idx[None, :] - idx[:, None]
        rel_clamped = rel.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        bias = self.rel_bias[rel_clamped].permute(2,0,1)[None]
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, T, C)
        return self.proj(out)

class MotionAmplifier(nn.Module):
    """
    Learnable motion amplification with residual connection.
    """
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.zeros(1))
        self.alpha_scale = nn.Parameter(torch.tensor(0.5))
        self.beta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        device = x.device
        alpha = self.alpha.to(device)
        beta = self.beta.to(device)
        alpha_scale = self.alpha_scale.clamp(0.0, 1.0).to(device)
        beta_scale = self.beta_scale.to(device)

        motion_strength = torch.norm(x, dim=-1, keepdim=True)
        scale = 1 + (alpha * alpha_scale) * motion_strength.clamp(max=5.0)
        bias = beta * beta_scale
        out = (x * scale + bias).clamp(-10.0, 10.0)
        return out + x  # residual

class DecoderLayer(nn.Module):
    """
    PreNorm decoder layer with RotaryAttention and SwiGLU FeedForward + DropPath.
    """
    def __init__(self, dim, hidden_dim, num_heads, qkv_bias, max_rel_dist, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = RotaryAttention(dim, num_heads, qkv_bias, max_rel_dist, dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, context=None):
        # Pre-Attn
        h = self.norm1(x)
        attn_out = self.self_attn(h, context)
        x = x + self.drop_path1(attn_out)
        # Pre-FFN
        h2 = self.norm2(x)
        a, b = self.fc1(h2).chunk(2, dim=-1)
        ffn_out = self.fc2(a * F.gelu(b))
        ffn_out = self.dropout_ffn(ffn_out)
        x = x + self.drop_path2(ffn_out)
        return x

class DecoderModel(nn.Module):
    """
    Optimized decoder: PreNorm, DropPath, RotaryAttention, SwiGLU FFN, MotionAmp.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        qkv_bias: bool = True,
        max_rel_dist: int = 30,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        motion_alpha: float = 0.3,
    ):
        super().__init__()

        self.input_dropout = nn.Dropout(dropout)
        self.input_map = nn.Linear(input_dim, output_dim)

        self.input_norm = nn.LayerNorm(output_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim=output_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                max_rel_dist=max_rel_dist,
                dropout=dropout,
                drop_path=dpr[i],
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(output_dim)
        self.output_map = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.motion_amp = MotionAmplifier(alpha=motion_alpha)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, input_dim)
            context: Style tokens or memory (B, T, output_dim)
        Returns:
            Amplified output (B, T, output_dim)
        """
        x = self.input_dropout(x)
        h = self.input_map(x)
        h = self.input_norm(h + context)
        for layer in self.layers:
            h = layer(h, context=context)
        h = self.norm(h)
        out = self.output_map(h)
        out = self.motion_amp(out)
        return out