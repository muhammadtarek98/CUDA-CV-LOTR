import torch
from torch.nn import Module
from .InterpolationChLayer import InterpolationChLayer

class InterpolationChModule(Module):
    def __init__(self,ch:int):
        super(InterpolationChModule, self).__init__()
        self.ch = ch

    def forward(self, input1:torch.Tensor, input2:torch.Tensor)->torch.Tensor:
        return InterpolationChLayer.apply(input1, input2)

