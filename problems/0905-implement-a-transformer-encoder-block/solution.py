import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp, dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        
        normed_x1 = self.norm1(x)
        attn_out = self.attn(normed_x1, normed_x1, normed_x1, need_weights=False)[0]

        x = x + self.dropout(attn_out)
   
        normed_x2 = self.norm2(x)
        mlp_out = self.mlp(normed_x2)
        x = x + self.dropout(mlp_out)

        return x

