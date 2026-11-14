#pragma once
#include <torch/torch.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight);

at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n);
template <typename scalar_t>
__global__ void devoxelize_forward_kernel(int N, int c,
                                          const int *__restrict__ indices,
                                          const scalar_t *__restrict__ weight,
                                          const scalar_t *__restrict__ feat,
                                          scalar_t *__restrict__ out);
template <typename scalar_t>
__global__ void devoxelize_backward_kernel(
    int N, int n, int c, const int *__restrict__ indices,
    const scalar_t *__restrict__ weight, const scalar_t *__restrict__ top_grad,
    scalar_t *__restrict__ bottom_grad);