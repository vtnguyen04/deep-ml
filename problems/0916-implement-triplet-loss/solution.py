import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    d_pos_sq = (anchor - positive).pow(2).sum(dim=-1)

    d_neg_sq = (anchor - negative).pow(2).sum(dim=-1)

    losses = F.relu(d_pos_sq - d_neg_sq + margin)

    return losses.mean()