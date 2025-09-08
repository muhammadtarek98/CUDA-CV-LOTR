# modules/FlowProjectionModule.py
from torch.nn import Module
from .FlowProjectionLayer import FlowProjectionLayer #, FlowFillholeLayer

class FlowProjectionModule(Module):
    def __init__(self, requires_grad:bool = True)->None:
        super(FlowProjectionModule, self).__init__()
        self.f = FlowProjectionLayer(requires_grad)

    def forward(self, input1:torch.Tensor)->torch.Tensor:
        return self.f(input1)
