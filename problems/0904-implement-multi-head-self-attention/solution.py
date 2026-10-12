import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            score += mask
        
        attn_weights = F.softmax(score, dim=-1)

        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(out)





        





