# modules/InterpolationLayer.py
import torch
from torch.nn import Module
from SeparableConv.SeparableConv.SeparableConvLayer import SeparableConvLayer

class SeparableConvModule(Module):
    def __init__(self,filter_size:int|tuple)->None:
        super(SeparableConvModule, self).__init__()
        self.f = SeparableConvLayer(filter_size)

    def forward(self, input1:torch.Tensor, input2:torch.Tensor, input3:torch.Tensor):
        return self.f(input1, input2, input3)

    #we actually dont need to write the backward code for a module, since we have 

