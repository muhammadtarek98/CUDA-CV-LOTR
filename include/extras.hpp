#pragma once
#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> dense_se3_forward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius);

std::vector<torch::Tensor> dense_se3_backward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius);


std::vector<torch::Tensor> se3_build_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius);


std::vector<torch::Tensor> se3_build_backward_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius);


std::vector<torch::Tensor> cholesky_solve6x6_forward_cuda(
  torch::Tensor H, torch::Tensor b);

std::vector<torch::Tensor> cholesky_solve6x6_backward_cuda(
  torch::Tensor H, torch::Tensor b, torch::Tensor dx);
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius);
std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius);
std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius);
std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) ;
std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius);
std::vector<torch::Tensor> se3_build(
    torch::Tensor attention,
    torch::Tensor transforms,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    int radius) ;
std::vector<torch::Tensor> se3_build_backward(
    torch::Tensor attention,
    torch::Tensor transforms,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    torch::Tensor H_grad,
    torch::Tensor b_grad,
    int radius);
std::vector<torch::Tensor> se3_build_inplace(
    torch::Tensor transforms,
    torch::Tensor embeddings,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    int radius);


std::vector<torch::Tensor> se3_build_inplace_backward(
    torch::Tensor transforms,
    torch::Tensor embeddings,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    torch::Tensor H_grad,
    torch::Tensor b_grad,
    int radius);
std::vector<torch::Tensor> cholesky6x6_forward(
    torch::Tensor H,
    torch::Tensor b);
std::vector<torch::Tensor> cholesky6x6_backward(
    torch::Tensor H,
    torch::Tensor b,
    torch::Tensor dx);