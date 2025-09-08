# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import SeparableConvFlow.separableconvflow_cuda as my_lib
import warnings
#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py
class SeparableConvFlowFunction(Function):
    @staticmethod
    def forward(ctx, input1:torch.Tensor, input2:torch.Tensor, input3:torch.Tensor, filter_size:int|tuple)->torch.Tensor:
        # Validate input dimensions
        intInputHeight = input1.size(2)
        intInputWidth = input1.size(3)
        intOutputHeight = min(input2.size(2), input3.size(2))
        intOutputWidth = min(input2.size(3), input3.size(3))
        assert intInputHeight - filter_size == intOutputHeight - 1
        assert intInputWidth - filter_size == intOutputWidth - 1
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        assert input3.is_contiguous()
        # Create output tensor
        flow_output = torch.zeros(input1.size(0), 2, intOutputHeight, intOutputWidth,
                                  device=input1.device, dtype=input1.dtype)
        # Call CUDA/CPU implementation
        if input1.device=='cuda':
            my_lib.SeparableConvFlowLayer_gpu_forward(input1, input2, input3, flow_output)
        else:
            my_lib.SeparableConvFlowLayer_cpu_forward(input1, input2, input3, flow_output)

        # Save tensors for backward pass
        ctx.save_for_backward(input1, input2, input3)
        ctx.filter_size = filter_size

        return flow_output

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor)->tuple:
        input1, input2, input3 = ctx.saved_tensors
        filtersize = ctx.filter_size
        # Initialize gradients
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)
        grad_input3 = torch.zeros_like(input3)
        # Call CUDA/CPU backward implementation
        if grad_output.is_cuda:
            my_lib.SeparableConvFlowLayer_gpu_backward(
                input1, input2, input3, grad_output, grad_input1, grad_input2, grad_input3
            )
        else:
            my_lib.SeparableConvFlowLayer_cpu_backward(
                input1, input2, input3, grad_output, grad_input1, grad_input2, grad_input3
            )

        return grad_input1, grad_input2, grad_input3, None
