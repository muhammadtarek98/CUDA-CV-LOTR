import torch
from torch.nn import Module
from .InterpolationLayer import InterpolationLayer

class InterpolationModule(Module):
    def __init__(self)->None:
        super(InterpolationModule, self).__init__()

    def forward(self, input1:torch.Tensor, input2:torch.Tensor)->torch.Tensor:
        return InterpolationLayer.apply(input1, input2)

    #we actually dont need to write the backward code for a module, since we have 

