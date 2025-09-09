import time
import torch
import numpy as np
from DepthFlowProjection.DepthFlowProjection import DepthFlowProjectionModule
from FilterInterpolation.FilterInterpolation import FilterInterpolationModule
from FlowProjection.FlowProjection import FlowProjectionModule
from Interpolation.Interpolation import InterpolationModule
from InterpolationCh.InterpolationCH import InterpolationChModule
from SeparableConv.SeparableConv import SeparableConvModule
from SeparableConvFlow.SeparableConvFlow import SeparableConvFlowModule


def test_SeparableConvFlowModule(input1, input2, input3, filtersize):
    FilterInterpolate = SeparableConvFlowModule(filtersize)

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = torch.autograd.Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    FilterInterpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    print("Check the forward path between CPU and GPU...", end='\t')
    x = (output_cuda - output.cuda()) * 2 / (torch.abs(output_cuda) + torch.abs(output).cuda())
    x = torch.max(torch.abs(x))

    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(output_cuda - output.cuda()))
    else:
        print("output pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = (input2_cuda.grad - input2.grad.cuda()) * 2 / (torch.abs(input2_cuda.grad) + torch.abs(input2.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = (input3_cuda.grad - input3.grad.cuda()) * 2 / (torch.abs(input3_cuda.grad) + torch.abs(input3.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    return t2 - t1, t3 - t2


def test_SeparableConvModule(input1, input2, input3, filtersize):
    FilterInterpolate = SeparableConvModule(filtersize)

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    temp = input1.grad

    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = torch.autograd.Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    FilterInterpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    print("Check the forward path between CPU and GPU...", end='\t')
    x = (output_cuda - output.cuda()) * 2 / (torch.abs(output_cuda) + torch.abs(output).cuda())
    x = torch.max(torch.abs(x))

    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')

    print("Check the backward path between CPU and GPU...", end='\t')
    x = (input1_cuda.grad - input1.grad.cuda()) * 2 / (torch.abs(input1_cuda.grad) + torch.abs(input1.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = (input2_cuda.grad - input2.grad.cuda()) * 2 / (torch.abs(input2_cuda.grad) + torch.abs(input2.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = (input3_cuda.grad - input3.grad.cuda()) * 2 / (torch.abs(input3_cuda.grad) + torch.abs(input3.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    return t2 - t1, t3 - t2


def test_FilterInterpolation(input1, input2, input3):
    FilterInterpolate = FilterInterpolationModule()

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = torch.autograd.Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    FilterInterpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")


    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')

    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))

    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = input3_cuda.grad - input3.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    return t2 - t1, t3 - t2


def test_InterpolationModule(input1, input2):
    Interpolate = InterpolationModule()

    t1 = time.time()

    output = Interpolate(input1, input2)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    output_cuda = Interpolate(input1_cuda, input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    return t2 - t1, t3 - t2


def test_InterpolationChModule(input1, input2):
    Interpolate = InterpolationChModule(input1.size(1))
    t1 = time.time()
    output = Interpolate(input1, input2)
    t2 = time.time()
    output.backward(output.data)
    t3 = time.time()
    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    output_cuda = Interpolate(input1_cuda, input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    return t2 - t1, t3 - t2


def test_FlowProjectionModule(input1):
    Project = FlowProjectionModule()
    t1 = time.time()
    output = Project(input1)
    t2 = time.time()
    output.backward(output.data)
    t3 = time.time()
    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    Project = FlowProjectionModule()  # regnenerate
    t1 = time.time()
    output_cuda = Project(input1_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass", end='\t')
    print("\n\n")
    return t2 - t1, t3 - t2


def test_DepthFlowProjectionModule(input1, input2):
    input1 = torch.autograd.Variable(input1.data.type(torch.FloatTensor), requires_grad=True)
    input2 = torch.autograd.Variable(input2.data.type(torch.FloatTensor), requires_grad=True)
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)

    Project = DepthFlowProjectionModule(input1_cuda.requires_grad)  # regnenerate
    output=Project(input1,input2)
    t1 = time.time()
    output_cuda = Project(input1_cuda, input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("\n\n")
    return t2 - t1, t3 - t2


def test_WeightedFlowProjectionModule(input1, input2, input3):
    Project = WeightedFlowProjectionModule(threshold=20.0 / 255.0, requires_grad=True)
    t1 = time.time()
    output = Project(input1, input2, input3)
    t2 = time.time()
    output.backward(output.data)
    t3 = time.time()
    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = torch.autograd.Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    Project = WeightedFlowProjectionModule(threshold=20.0 / 255.0, requires_grad=True)  # regnenerate
    t1 = time.time()
    output_cuda = Project(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
    else:
        print("pass", end='\n')
    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass", end='\t')
    print("\n\n")
    return t2 - t1, t3 - t2


def test_AdaptiveWeightInterpolationModule(input1, input2, input3, input4):
    training = True
    Interpolate = AdaptiveWeightInterpolationModule(training=training)
    t1 = time.time()
    output = Interpolate(input1, input2, input3, input4)
    t2 = time.time()
    if training:
        grad = output.data
        output.backward(grad)
        print(input3.grad)
    t3 = time.time()
    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    temp = input1.grad
    input1_cuda = torch.autograd.Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = torch.autograd.Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = torch.autograd.Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input4_cuda = torch.autograd.torch.autograd.Variable(input4.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    Interpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = Interpolate(input1_cuda, input2_cuda, input3_cuda, input4_cuda)
    t2 = time.time()
    if training:
        grad = output_cuda.data
        output_cuda.backward(grad)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    print("Check the forward path between CPU and GPU...", end='\n')
    x = output_cuda - output.cuda()
    print("==>total number of difference")
    print(torch.sum(torch.abs(x) > 1e-6))

    x = torch.max(torch.abs(x))
    print("==>max difference value is ")
    print(x)
    print(torch.sum(output_cuda > 1))
    print(torch.sum(output.cuda() > 1))

    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)

    else:
        print("pass", end='\n')

    if not training:
        return t2 - t1, t3 - t2

    print("Check the backward path between CPU and GPU...", end='\t')
    y = input1_cuda.grad - input1.grad.cuda()
    x = y.cpu().data.numpy()
    x = torch.max(torch.abs(y))
    print(x)

    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = input3_cuda.grad - input3.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')
    x = input4_cuda.grad - input4.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if x.cpu().data.numpy()[0] > 1e-6:
        print(x)
        print(torch.mean(input4_cuda.grad - input4.grad.cuda()))
    else:
        print("pass", end='\n')

    return t2 - t1, t3 - t2


B, C, H, W = 1, 2, 512, 704
input1 = torch.autograd.Variable(torch.arange(0.0, B * C * H * W).view(B, C, H, W), requires_grad=True)
input3 = torch.autograd.Variable(torch.arange(0.0, B * 3 * H * W).view(B, 3, H, W), requires_grad=True)
input2 = torch.autograd.Variable(torch.arange(0.0, B * 1 * H * W).view(B, 1, H, W), requires_grad=True)

ftimes = []
btimes = []
for i in range(10):
    input1.data.uniform_(-1.0, 1.0)
    input2.data.uniform_(0.1, 1.0)  # must be larger than zero
    input1 = torch.autograd.Variable(input1.clone().data, requires_grad=True)  # to delete the graph in InterpolationModule
    input2 = torch.autograd.Variable(input2.clone().data, requires_grad=True)
    ftime, btime = test_DepthFlowProjectionModule(input1, input2)
    ftimes.append(ftime)
    btimes.append(btime)
print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) + "s\t" + str(
    numpy.array(btimes).mean()) + "s\t\n\n\n\n")