# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hubert.modeling_hubert import HubertModel
from sty_models import StyleEncoder
from decoder import DecoderModel
import numpy as np
from transformers import HubertModel

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


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B, T, C)
        t = torch.arange(x.size(1), device=x.device).float()  # [0,1,...,T-1]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (T, C/2)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (T, C)
        return emb.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, T, C)


class PersonaDiff(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim,
            latent_dim=512,
            audio_feature_dim=1536,
            diffusion_steps=500,
    ) -> None:
        super().__init__()
        self.i_fps = args.input_fps
        self.o_fps = args.output_fps
        self.device = args.device
        self.one_hot_timesteps = np.eye(args.diff_steps)

        self.input_projection = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.audio_encoder_content = HubertModel.from_pretrained(
            " "
        )
        self.audio_dim = self.audio_encoder_content.encoder.config.hidden_size
        self.audio_encoder_content.feature_extractor._freeze_parameters()

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.1)
        )

        self.pos_emb = SinusoidalPosEmb(latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
            nn.LayerNorm(latent_dim)
        )

        self.cross_attn_norm = nn.LayerNorm(latent_dim )
        self.cross_proj = nn.Linear(latent_dim * 3, latent_dim)

        self.fusion_dropout = nn.Dropout(p=0.2)

        self.style_encoder = StyleEncoder(args, vertice_dim, latent_dim)
        self.style_dropout = nn.Dropout(0.1)

        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim , num_heads=8, batch_first=True, dropout=0.1)

        self.DecoderModel = DecoderModel(
            input_dim=latent_dim,
            output_dim=latent_dim,
            hidden_dim=512
        )


        self.final_layer = nn.Linear(latent_dim, vertice_dim)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        self.obj_vector = nn.Linear(
            len(args.train_subjects.split()),
            latent_dim,
            bias=False
        )

    def forward(self, vertice, x_t, times, audio, template, one_hot):
        device = vertice.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        obj_embedding = self.obj_vector(one_hot)

        x_t = x_t.permute(1, 0, 2)  # [T,B,C]
        x_t = self.input_projection(x_t)
        x_t = x_t.permute(1, 0, 2)  # [B,T,C]

        motion = vertice - template.unsqueeze(1)

        aud_feat = self.audio_encoder_content(audio).last_hidden_state  # [B,T,D]
        aud_feat, x_t, frame_num = adjust_input_representation(
            aud_feat, x_t, self.i_fps, self.o_fps
        )
        aud_feat = aud_feat.permute(1, 0, 2)
        aud_feat = self.audio_projection(aud_feat).permute(1, 0, 2)  # [B,T,C]

        style_feat = self.style_encoder(motion, audio)  # [B,C]
        style_tokens = style_feat.unsqueeze(1).expand(-1, frame_num, -1)
        style_tokens = self.style_dropout(style_tokens)

        x_t = x_t[:, :frame_num]

        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2) #[B,T,C]

        combined = torch.cat([
            aud_feat,
            x_t,
            t_tokens
        ], dim=-1)
        combined = self.fusion_dropout(combined)
        x = self.cross_proj(combined)
        combined,_ = self.cross_attn(x,x, x)
        combined = combined + x
        combined = self.cross_attn_norm(combined)


        output = self.DecoderModel(
            x=combined,
            context=style_tokens
        )

        output = output * obj_embedding.unsqueeze(1)
        output = self.final_layer(output) + template.unsqueeze(1)

        return output


