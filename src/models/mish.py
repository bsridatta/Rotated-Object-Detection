import torch
import torch.nn.functional as F


@torch.jit.script
def mish(input):
    """
    Source: https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return input * torch.tanh(F.softplus(input))


class Mish(torch.nn.Module):
    """
    Source: https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py

    Applies the mish function element-wise:
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)
