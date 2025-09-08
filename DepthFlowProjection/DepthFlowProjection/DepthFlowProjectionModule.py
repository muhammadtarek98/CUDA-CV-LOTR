# modules/FlowProjectionModule.py
import torch
from torch.nn.modules.module import Module
from .DepthFlowProjectionLayer import DepthFlowProjectionLayer #, FlowFillholeLayer

__all__ =['DepthFlowProjectionModule']

class DepthFlowProjectionModule(Module):
    def __init__(self, requires_grad :bool= True)->None:
        super(DepthFlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad
    def forward(self, input1:torch.Tensor, input2:torch.Tensor)->torch.Tensor:
        return DepthFlowProjectionLayer.apply(input1, input2,self.requires_grad)
