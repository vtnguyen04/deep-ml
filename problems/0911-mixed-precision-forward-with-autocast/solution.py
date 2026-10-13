import torch

def forward_bf16(model, x):
    # TODO: run model(x) inside torch.amp.autocast with dtype=bfloat16; return float32 output
    
    device_type = x.device.type

    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

        out = model(x)
    
    return out.float()