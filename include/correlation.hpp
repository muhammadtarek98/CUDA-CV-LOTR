#pragma once
#include <bits/stdc++.h>
#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
std::vector<torch::Tensor> corr_forward(torch::Tensor &fmap1,torch::Tensor &fmap2,torch::Tensor &coords,const int &radius);
std::vector<torch::Tensor> corr_backward(torch::Tensor &fmap1,torch::Tensor &fmap2,torch::Tensor &coords,torch::Tensor &corr_grad,int &radius);
std::vector<torch::Tensor> corr_cuda_forward(const torch::Tensor &fmap1,const torch::Tensor &fmap2,const torch::Tensor &coords,const int &radius);
std::vector<torch::Tensor> corr_cuda_backward(const torch::Tensor &fmap1,const torch::Tensor &fmap2,const torch::Tensor &coords,const torch::Tensor &corr_grad,const int &radius);