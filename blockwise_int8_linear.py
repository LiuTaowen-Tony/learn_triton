from functools import partial

import torch
import torch.nn as nn

from blockwise_int8_matmul_dequantise import (
    int8_matmul_block64_rowwise_dequantize,
    quantize_block_rowwise
)

class blockwise_int8_linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        X = X_3D.view(-1, X_3D.size(-1))
        X_int8, state_X = quantize_block_rowwise(X)
        W_int8, state_W = quantize_block_rowwise(W)

        # ctx.save_for_backward = X_int8, state_X, W_int8, state_W
        ctx.save_for_backward = X, W

        return int8_matmul_block64_rowwise_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx: torch.Any, G_3D) -> torch.Any:
        G = G_3D.reshape(-1, G_3D.size(-1))
        grad_X = grad_W = grad_bias = None

        X, W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_block_rowwise(G)
            W_int8, state_W = quantize_block_rowwise(W)
            grad_X = int8_matmul_block64_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

class BlockInt8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = torch.float16,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

    def prepare_for_eval(self):
        W_int8, state_W = quantize_block_rowwise(self.weight)
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

        del self.weight

    def forward(self, x):
        if self.training:
            return blockwise_int8_linear.apply(x, self.weight, self.bias)
        else:
            if not hasattr(self, "W_int8"):
                return blockwise_int8_linear.apply(x, self.weight, self.bias)

            X = x.view(-1, x.size(-1))
            X_int8, state_X = quantize_block_rowwise(X)

            return int8_matmul_block64_rowwise_dequantize(
                X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
            ).view(*x.size()[:-1], -1)
