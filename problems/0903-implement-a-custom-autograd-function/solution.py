import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        ctx.save_for_backward(x)

        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        x, = ctx.saved_tensors

        grad_input = grad_output * (x > 0).to(grad_output.dtype)

        return grad_input


