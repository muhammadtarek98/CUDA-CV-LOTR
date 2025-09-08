# modules/FlowProjectionModule.py
import torch
from torch.nn.modules.module import Module
from .minDepthFlowProjectionLayer import minDepthFlowProjectionLayer
__all__ =['minDepthFlowProjectionModule']

class minDepthFlowProjectionModule(Module):
    def __init__(self, requires_grad:bool = True)->None:
        super(minDepthFlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad
    def forward(self, input1, input2:torch.Tensor):
        return minDepthFlowProjectionLayer.apply(input1, input2,self.requires_grad)
