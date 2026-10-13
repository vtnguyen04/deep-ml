import torch
import torch.nn as nn

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    
    optimizer.zero_grad()

    total_loss = 0.0

    num_batches = len(micro_batches)

    for x, y in micro_batches:
        outputs = model(x)

        loss = criterion(outputs, y)
        total_loss += loss.item()
        scaled_loss = loss / num_batches

        scaled_loss.backward()

    
    optimizer.step()

    return total_loss / num_batches
