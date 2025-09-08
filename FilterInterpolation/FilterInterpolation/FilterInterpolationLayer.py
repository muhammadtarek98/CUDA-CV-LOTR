# this is for wrapping the customized layer
import torch
from torch.autograd import Function
from FilterInterpolation import filterinterpolation_cuda as my_lib
#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class FilterInterpolationLayer(Function):
    def __init__(self):
        super(FilterInterpolationLayer,self).__init__()
    @staticmethod
    def forward(ctx, input1:torch.Tensor,input2:torch.Tensor,input3:torch.Tensor)->torch.Tensor:
        assert(input1.is_contiguous())
        assert(input2.is_contiguous())
        assert (input3.is_contiguous())
        if input1.is_cuda :
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        else:
            output = torch.FloatTensor(input1.data.size())
            my_lib.FilterInterpolationLayer_cpu_forward(input1, input2, input3, output)
        ctx.save_for_backward(input1, input2,input3)
        return output

    @staticmethod
    def backward(ctx, gradoutput:torch.Tensor)->tuple:
        input1, input2, input3= ctx.saved_tensors
        gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()
        gradinput3 = torch.cuda.FloatTensor().resize_(input3.size()).zero_()
        if input1.is_cuda:
            err = my_lib.FilterInterpolationLayer_gpu_backward(input1,input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        else:
            err = my_lib.FilterInterpolationLayer_cpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        return gradinput1, gradinput2,gradinput3

# calculate the weights of flow         
class WeightLayer(Function):
    def __init__(self, lambda_e:float= 10.0/255.0, lambda_v:float= 1.0, Nw:int=3)->None:
        super(WeightLayer,self).__init__()
        self.lambda_e = lambda_e
        self.lambda_v = lambda_v
        self.Nw = Nw
    def forward(self, input1:torch.Tensor,input2:torch.Tensor,input3:torch.Tensor)->torch.Tensor:
        self.input1 = input1.contiguous() # ref1 image
        self.input2 = input2.contiguous() # ref2 image
        self.input3 = input3.contiguous()
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        output =  torch.zeros(input1.size(0), 1 , input1.size(2), input1.size(3))
        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.WeightLayer_gpu_forward(input1, input2, input3,output,self.lambda_e,  self.lambda_v, self.Nw)
            if err != 0 :
                print(err)
        else:
            err = my_lib.WeightLayer_cpu_forward(input1, input2, input3,  output, self.lambda_e ,  self.lambda_v, self.Nw)
            if err != 0 :
                print(err)
        self.output = output
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput:torch.Tensor)->tuple:
        gradinput1 = torch.zeros(self.input1.size())
        gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        if self.input1.is_cuda:
            gradinput1 = gradinput1.cuda(self.device)
            gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            err = my_lib.WeightLayer_gpu_backward(self.input1,self.input2,self.input3, self.output,gradoutput,
                gradinput1, gradinput2, gradinput3,self.lambda_e,  self.lambda_v, self.Nw)
            if err != 0 :
                print(err)
        else:
            err = my_lib.WeightLayer_cpu_backward(self.input1, self.input2,self.input3, self.output,
                                                  gradoutput,gradinput1, gradinput2, gradinput3, self.lambda_e, self.lambda_v, self.Nw )
            if err != 0 :
                print(err)
        return gradinput1, gradinput2, gradinput3
  
class PixelValueLayer(Function):
    def __init__(self, sigma_d:int=3, tao_r:int= 0.05, Prowindow:int=2)->None:
        super(PixelValueLayer,self).__init__()
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow
    def forward(self, input1:torch.Tensor, input3:torch.Tensor, flow_weights:torch.Tensor)->torch.Tensor:
        self.input1 = input1.contiguous() # ref1 image
        self.input3 = input3.contiguous() # ref1 flow
        self.flow_weights = flow_weights.contiguous() # ref1 flow weights
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        output = torch.zeros(input1.size())
        if input1.is_cuda:
            output = output.cuda()            
            err = my_lib.PixelValueLayer_gpu_forward(input1,input3,flow_weights,output,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelValueLayer_cpu_forward(input1,  input3, flow_weights, output,self.sigma_d,    self.tao_r ,  self.Prowindow)
            if err != 0 :
                print(err)
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput:torch.Tensor)->tuple:
        gradinput1 = torch.zeros(self.input1.size())
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())
        if self.input1.is_cuda:
            gradinput1 = gradinput1.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)
            err = my_lib.PixelValueLayer_gpu_backward(self.input1,self.input3, self.flow_weights,gradoutput,gradinput1,
                                                      gradinput3,gradflow_weights,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelValueLayer_cpu_backward(self.input1,self.input3, self.flow_weights,
                gradoutput,gradinput1,gradinput3, gradflow_weights,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        return gradinput1,  gradinput3, gradflow_weights

class PixelWeightLayer(Function):
    def __init__(self,threshold:float, sigma_d:int=3, tao_r:float=0.05, Prowindow:int=2)->None:
        super(PixelWeightLayer,self).__init__()
        self.threshold  = threshold
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3:torch.Tensor, flow_weights:torch.Tensor)->torch.Tensor:
        self.input3 = input3.contiguous() # ref1 flow
        self.flow_weights = flow_weights.contiguous() # ref1 flow weights
        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)])
        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.PixelWeightLayer_gpu_forward(input3, flow_weights,output,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelWeightLayer_cpu_forward(input3, flow_weights, output, self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        self.output = output
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput:torch.Tensor)->tuple:
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())
        if self.input3.is_cuda:
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)
            err = my_lib.PixelWeightLayer_gpu_backward( self.input3,self.flow_weights,self.output,
                gradoutput,gradinput3, gradflow_weights,self.threshhold,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelWeightLayer_cpu_backward(self.input3, self.flow_weights, self.output,gradoutput,
                gradinput3, gradflow_weights, self.threshhold,self.sigma_d,self.tao_r ,self.Prowindow)
            if err != 0 :
                print(err)
        return gradinput3, gradflow_weights

class ReliableWeightLayer(Function):
    def __init__(self, threshold, sigma_d:int=3, tao_r:float=0.05, Prowindow:int=2)->None:
        super(ReliableWeightLayer,self).__init__()
        self.threshold = threshold
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3:torch.Tensor)->torch.Tensor:
        self.input3 = input3.contiguous() # ref1 flow
        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)] )
        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.ReliableWeightLayer_gpu_forward(input3, output,self.sigma_d,self.tao_r ,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.ReliableWeightLayer_cpu_forward(input3, output,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        self.output= output
        return output
    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput:torch.Tensor)->torch.Tensor:
        gradinput3 = torch.zeros(self.input3.size())
        if self.input3.is_cuda:
            gradinput3 = gradinput3.cuda(self.device)
            err = my_lib.ReliableWeightLayer_gpu_backward(self.input3,self.output,gradoutput, gradinput3,self.threshhold,
                                                          self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.ReliableWeightLayer_cpu_backward(self.input3, self.output,gradoutput,gradinput3,
                self.threshold,self.sigma_d,self.tao_r,self.Prowindow)
            if err != 0 :
                print(err)
        return gradinput3