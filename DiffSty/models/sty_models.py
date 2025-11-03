import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from wav2vec import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from ikan.GroupKAN import GroupKAN
from fused import ContentAudioFusion

class AddAndNorm(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, residual):
        return self.norm(x + residual)
class Gate(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.gate = nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x) * self.sigmoid(self.gate(x))

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        try:
            return x.view(*self.shape)
        except RuntimeError:
            new_shape = list(self.shape)
            new_shape[0] = -1
            return x.view(*new_shape)

class GRN_KAN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gate = Gate(latent_dim)
        self.kan = GroupKAN(
             layers_hidden=[512,512],
             act_mode="swish",
             drop=0.3
        )
        self.add_norm = AddAndNorm(latent_dim)

    def forward(self, x):
        residual = x
        gate_out = self.gate(x)
        kan_out = self.kan(x)
        processed = gate_out + kan_out
        return self.add_norm(processed, residual)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, T, _ = x.shape
        x = x + self.pe[:T].transpose(0,1)
        return self.dropout(x)


def apply_mask(tensor, mask_tensor):
    while mask_tensor.dim() < tensor.dim():
        mask_tensor = mask_tensor.unsqueeze(-1)
    return tensor * mask_tensor.to(tensor.dtype)


class TGAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment=8,
                 reduction_ratio=16,
                 kernel_size=3,
                 dropout_path=0.15):
        super().__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.reduction = reduction_ratio
        self.dropout_path = dropout_path

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k,
                          padding=k // 2, groups=in_channels, bias=False),
                nn.GroupNorm(4, in_channels)
            )
            for k in [3, 5, 7]
        ])

        self.scale_weights = nn.Parameter(torch.zeros(3))

        self.temporal_gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // self.reduction, 1, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // self.reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.res_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.GroupNorm(4, in_channels),
            nn.Dropout(0.2)
        )
        self.res_norm = nn.GroupNorm(4, in_channels)

    def forward(self, x):
        B, C, T = x.shape

        x_spatial = x.unsqueeze(-1)  # (B,C,T,1)

        spatial_mask = self.spatial_attn(x_spatial)  # (B,1,T,1)
        x_attn = x_spatial * spatial_mask

        scales = []
        for conv in self.multi_scale:
            scale = conv(x_attn).squeeze(-1)  # (B,C,T)
            scales.append(scale)
        fused = torch.stack(scales, dim=-1)  # (B,C,T,3)
        weights = F.softmax(self.scale_weights, dim=0)
        fused = (fused * weights).sum(dim=-1)  # (B,C,T)

        if self.training and torch.rand(1) < self.dropout_path:
            residual = torch.zeros_like(fused)
        else:
            residual = self.res_conv(x)
            residual = self.res_norm(residual)

        fused = fused * torch.bernoulli(
            torch.ones(B, C, 1, device=x.device) * (1 - self.dropout_path)
        )

        ctx = fused.mean(dim=2, keepdim=True)  # (B,C,1)
        gate = self.temporal_gate(ctx)  # (B,C,1)

        out = fused * gate + residual
        return out


class MaskedInstanceNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, channels))
        self.shift = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x, mask):
        # x: (B, T, C), mask: (B, T)
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        x_perm = x  # (B, C, T)
        sum_x = (x_perm * mask).sum(dim=2, keepdim=True)  # (B, C, 1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=self.eps)
        mean = sum_x / count
        var = ((x_perm - mean)**2 * mask).sum(dim=2, keepdim=True) / count
        min_var = torch.full_like(var, 0.1)
        var = torch.maximum(var, min_var)
        std = torch.sqrt(var)
        x_norm = (x_perm - mean) / std
        x_norm = x_norm * self.scale + self.shift
        out = x_norm.permute(0, 2, 1)  # (B, T, C)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.permute(0, 2, 1)
        return out

def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    elif ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


class MSBlock(nn.Module):
    def __init__(self, in_ch, kernel_size):
        super().__init__()

        self.kan = GRN_KAN(in_ch)
        self.body = nn.Sequential(
            TGAM(in_ch, kernel_size=kernel_size),
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GELU(),
            nn.GroupNorm(1, in_ch),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_ch, in_ch // 16, 1),
            nn.GELU(),
            nn.Conv1d(in_ch // 16, in_ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        res = x
        x = self.body(x)
        out = self.kan(x.transpose(1, 2)).transpose(1, 2)
        ca  = self.se(out) #[1, 256, 1]
        out = out * ca + res

        return out


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
    def forward(self, x):  # x: [B, T, C]
        y = x.permute(0, 2, 1)   # [B, C, T]
        y = self.conv(y)
        y = y.permute(0, 2, 1)   # [B, T, C]
        return y


class TransformerEncoderLayerWithRelPos(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.nhead   = nhead
        self.head_dim = d_model // nhead
        self.max_len = max_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.rel_pos_bias = nn.Parameter(torch.zeros(2*max_len-1))

    def forward(self, x, mask=None):

        B, T, C = x.shape
        device = x.device

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim)  # [B, T, H, D]
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # scaled_dot = (q @ k.transpose(-2, -1)) / sqrt(d_k)  -> [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]

        idxs = torch.arange(T, device=device)
        rel_idx = idxs.unsqueeze(1) - idxs.unsqueeze(0)
        rel_idx = rel_idx + (self.max_len - 1)           # shift to [0..2*max_len-2]
        rel_idx = rel_idx.clamp(0, 2*self.max_len-2)     # clip out-of-range
        # rel_pos_bias_vec: [2*max_len-1], rel_idx: [T, T]
        pos_bias = self.rel_pos_bias[rel_idx]  # [T, T]
        pos_bias = pos_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        scores = scores + pos_bias

        if mask is not None:

            mask_k = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask_k == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)   # [B, H, T, D]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, d_model]
        attn_out = self.proj_dropout(self.out_proj(attn_out))
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class  StyleEncoder(nn.Module):
    def __init__(self, args,vertice_dim,latent_dim):
        super().__init__()
        self.args = args

        self.motion_map = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim )
            # nn.GELU(),
            # nn.Dropout(0.3),
            # nn.Linear(latent_dim * 2, latent_dim),
        )
        nn.init.kaiming_normal_(self.motion_map[0].weight, mode='fan_in', nonlinearity='linear')

        # multi-scale modules
        self.multiscale = nn.ModuleList([
            MSBlock(latent_dim, k) for k in [3, 5]
        ])
        self.tau = nn.Parameter(torch.tensor(1.0))
        nn.init.constant_(self.tau, 1.0)

        self.res_conv = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1) for _ in range(3)
        ])

        self.fusion_weights = nn.Parameter(torch.zeros(3))
        self.multiscale_norm = MaskedInstanceNorm(latent_dim)

        self.num_layers = 2
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=latent_dim * 2,
                dropout=0.3,
                batch_first=True,
                activation='gelu',
                norm_first=True
            ) for _ in range(self.num_layers)
        ])

        self.conv_layers = nn.ModuleList([
            DepthwiseConv1d(channels=latent_dim, kernel_size=3, padding=1)
            for _ in range(self.num_layers)
        ])

        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim, eps=1e-6) for _ in range(self.num_layers)
        ])
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim, eps=1e-6) for _ in range(self.num_layers)
        ])

        # self.final_selector = FeatureSelector(latent_dim,use_lstm=False)

        self.PE = PositionalEncoding(latent_dim)
        self.selector_norm = nn.LayerNorm(latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)


        self.audio_encoder_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
            " "
        )
        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            " "
        )
        self.audio_encoder_emotion.wav2vec2.feature_extractor._freeze_parameters()
        # self.audio_proj = nn.Linear(2048, latent_dim)

        self.audio_proj = nn.Sequential(
            nn.Linear(2048, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.fusion_blocks = ContentAudioFusion(
            embed_dim = latent_dim,
            num_heads = 4,

        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.3)

        self.final_lnf = nn.LayerNorm(latent_dim)


    def forward(self, motion, audio_input):
        B, T, _ = motion.size()

        seq_lengths = torch.full((B,), T, dtype=torch.long, device=motion.device)
        mask = torch.arange(T, device=motion.device)[None, :] < seq_lengths[:, None] #[B, T]

        motion = motion.permute(1, 0, 2)
        base_feat = self.motion_map(motion).permute(1, 0, 2) # [B, T, 512]

        ms_outputs = []
        x = base_feat.permute(0,2,1).contiguous()
        for i, block in enumerate(self.multiscale):
            residual = x
            residual = self.res_conv[i](x)
            out = block(x) + residual
            ms_outputs.append(out.permute(0,2,1).contiguous())
        w = torch.softmax(self.fusion_weights / self.tau, dim=0)
        fused = sum(wi * mi for wi, mi in zip(w, ms_outputs)) #[B, T, 256]
        fused = self.multiscale_norm(fused, mask)

        pe = self.PE(fused)* mask.unsqueeze(-1)

        x = pe
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](x)
            x = self.attn_norms[i](attn_out + x)
            conv_out = self.conv_layers[i](x)
            x = self.conv_norms[i](conv_out + x)
        content_code = apply_mask(x, mask)
        content_code = self.output_norm(content_code)

        inputs = audio_input.to(self.audio_encoder_emotion.device)
        inputs = self.audio_feature_extractor(
            inputs.squeeze().cpu().numpy(),
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        ).input_values.to(audio_input.device)
        outputs = self.audio_encoder_emotion(inputs, output_hidden_states=True)
        audio_features = outputs.hidden_states[-1]  # (72,1024)
        audio_features = audio_features.unsqueeze(0)  # （B，T，C）

        audio_feat, x_t, frame_num = adjust_input_representation(audio_features, base_feat, 50,
                                                                 25)  # (1,T_face,1536) (1,T_face,512)
        audio_feat = audio_feat[:, :frame_num]
        content_code = content_code[:, :frame_num]
        audio_feat = self.audio_proj(audio_feat)  # (179,256)

        fused_style = self.fusion_blocks(content_code, audio_feat)  # （1，36，512） （1，36，512）

        return fused_style


