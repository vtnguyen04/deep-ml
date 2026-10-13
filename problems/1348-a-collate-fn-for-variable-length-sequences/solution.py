import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch, pad_value=0):
    # TODO: split the batch into sequences and labels
    # TODO: compute lengths and the batch's max length
    # TODO: build the padded (B, T_max) tensor filled with pad_value
    # TODO: build the (B, T_max) bool mask, True at real tokens
    # TODO: return (padded, lengths, mask, labels)
    sequences, labels = zip(*batch)

    B = len(batch)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_value)

    T_max = padded.size(1)

    mask = torch.arange(T_max, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, mask, labels
