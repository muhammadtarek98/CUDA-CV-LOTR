#pragma once
#include <torch/torch.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <THC/THCAtomics.cuh>
#include <cmath>
void to_dense_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                           const at::Tensor range, at::Tensor outputs);

void to_dense_backward_cuda(const at::Tensor top_grad,
                            const at::Tensor idx, const at::Tensor range,
                            const at::Tensor bottom_grad);

template <typename scalar_t>
__global__ void to_dense_forward_kernel(int N, int c, const scalar_t *__restrict__ feats, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ out);

template <typename scalar_t>
__global__ void to_dense_backward_kernel(int N, int c, const scalar_t *__restrict__ top_grad, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ bottom_grad);