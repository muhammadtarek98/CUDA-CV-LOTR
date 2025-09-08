# modules/InterpolationLayer.py
from torch.nn import Module
from SeparableConvFlow.SeparableConvFlow import SeparableConvFlowFunction
import torch
import warnings


class SeparableConvFlowModule(torch.nn.Module):
    def __init__(self, filter_size:int|tuple)->None:
        super(SeparableConvFlowModule, self).__init__()
        self.filter_size = filter_size
        warnings.warn("\nSeparable Conv Flow Layer is not precise enough for optical flow due to a division operation")

    def forward(self, input1:torch.Tensor, input2:torch.Tensor, input3:torch.Tensor)->torch.Tensor:
        # Pass filter_size as an argument to the Function
        return SeparableConvFlowFunction.apply(input1, input2, input3, self.filter_size)
