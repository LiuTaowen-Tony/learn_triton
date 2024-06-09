import torch
from blockwise_int8_matmul_dequantise import quantize_block_rowwise, int8_matmul_block64_rowwise_dequantize
from torch import nn

class _switchback_vectorrize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))

        ctx.save_for_backward = X, W
        # rowwise quantize for X
        # columnwise quantize for W (first rowwise, transpose later)
        X_int8, state_X = quantize_block_rowwise(X)
        W_int8, state_W = quantize_block_rowwise(W)

        # matmult, fused dequant and add bias
        # call kernel which expects rowwise quantized X and W
        return int8_matmul_block64_rowwise_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):
        X, W = ctx.save_for_backward

        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        G_int8, state_G = quantize_block_rowwise(G)
        W_int8, state_W = quantize_block_rowwise(W)
        if ctx.needs_input_grad[0]:
            # rowwise quantize for G, columnwise quantize for W and fused transpose
            # we call .t() for weight later because only A @ B^T is supported
            grad_X = int8_matmul_block64_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            # backward pass uses standard weight grad
            G_t_int8, state_G_t = quantize_block_rowwise(G.t())
            X_t_int8, state_X_t = quantize_block_rowwise(X.t())
            grad_W = int8_matmul_block64_rowwise_dequantize(
                G_t_int8, X_t_int8.t(), state_G_t, state_X_t, None)
            # grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

    

class Int8BlockLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            device=None,
            dtype=None,
        ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # By default, we use the global quantization.
        self._fn = _switchback_vectorrize


    def forward(self, x):
        return self._fn.apply(x, self.weight, self.bias)