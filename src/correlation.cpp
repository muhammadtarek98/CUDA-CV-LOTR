#include <torch/extension.h>
#include <bits/stdc++.h>
#include "correlation.hpp"


std::vector<torch::Tensor> corr_forward(torch::Tensor &fmap1,torch::Tensor &fmap2,torch::Tensor &coords,const int &radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return corr_cuda_forward(fmap1, fmap2, coords, radius);
}


std::vector<torch::Tensor> corr_backward(torch::Tensor &fmap1,torch::Tensor &fmap2,torch::Tensor &coords,torch::Tensor &corr_grad,int &radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return corr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


