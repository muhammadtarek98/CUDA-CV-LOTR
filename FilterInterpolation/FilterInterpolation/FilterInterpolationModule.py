# modules/AdaptiveInterpolationLayer.py
from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from .FilterInterpolationLayer import FilterInterpolationLayer,WeightLayer, PixelValueLayer,PixelWeightLayer,ReliableWeightLayer

class FilterInterpolationModule(Module):
    def __init__(self)->None:
        super(FilterInterpolationModule, self).__init__()

    def forward(self, input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)

class AdaptiveWeightInterpolationModule(Module):
    def __init__(self, training:bool= False, threshold:float=1e-6, lambda_e:float=30.0/255.0,
                 lambda_v:float=1.0, Nw:float=3.0,sigma_d:float=1.5,  tao_r:float=0.05,
                 Prowindow:int=2)->None:
        super(AdaptiveWeightInterpolationModule, self).__init__()
        self.calc_weight1 = WeightLayer(lambda_e, lambda_v, Nw )
        self.padder1 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.interpolate1 = PixelValueLayer(sigma_d, tao_r , Prowindow)
        self.interpolate1_1 = PixelWeightLayer(101 * threshold, sigma_d, tao_r, Prowindow)
        self.interpolate_R1_1 = ReliableWeightLayer(101 * threshold, sigma_d, tao_r, Prowindow)

        self.calc_weight2 = WeightLayer(lambda_e, lambda_v,Nw)
        self.padder2 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.interpolate2 = PixelValueLayer(sigma_d, tao_r , Prowindow )
        self.interpolate2_1 = PixelWeightLayer(101 * threshold, sigma_d, tao_r, Prowindow)
        self.interpolate_R2_1 = ReliableWeightLayer(101 * threshold, sigma_d, tao_r, Prowindow)
        self.training = training
        self.threshold = threshold
        return
    def forward(self, input1, input2, input3, input4):
        epsilon = 1e-6
        flow_weight1 = self.calc_weight1(input1,input2,input3 )
        p1 = self.interpolate1(input1, input3, flow_weight1)
        p1_r,p1_g,p1_b = torch.split(tensor=p1,split_size_or_sections=1,dim=1)
        pw1 = self.interpolate1_1(input3, flow_weight1)
        i1_r,i1_g,i1_b = (p1_r)/(pw1+self.threshold),(p1_g)/(pw1+self.threshold),(p1_b)/(pw1+self.threshold)
        r1 = pw1
        rw1 = self.interpolate_R1_1(input3)
        w1 = (r1)/(rw1+self.threshold)
        flow_weight2 = self.calc_weight2(input2,input1,input4)
        p2 = self.interpolate2(input2, input4, flow_weight2)
        p2_r,p2_g,p2_b = torch.split(tensor=p2,split_size_or_sections=1,dim=1)
        pw2 = self.interpolate2_1(input4, flow_weight2)
        i2_r,i2_g,i2_b = (p2_r)/(pw2+self.threshold),(p2_g)/(pw2+self.threshold),(p2_b)/(pw2+self.threshold)
        r2 = pw2
        rw2 = self.interpolate_R2_1(input4)
        w2 = (r2)/(rw2+self.threshold)
        w = w1+w2
        i_r = (i1_r * w1 + i2_r * w2)/ (w + self.threshold)
        i_g = (i1_g * w1 + i2_g * w2)/ (w + self.threshold)
        i_b = (i1_b * w1 + i2_b * w2)/ (w + self.threshold)
        if not self.training:
            i_r[w<= 10*self.threshold], i_g[w<=10*self.threshold], i_b[w<=10*self.threshold] = 0,0,0
            w[w <= 10 *self.threshold] = 0
        i = torch.cat(tensors=(i_r,i_g,i_b),dim=1)
        return i
