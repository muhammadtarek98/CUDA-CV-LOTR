#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include<torch/torch.h>
__global__ void hash_kernel(int N, const int *__restrict__ data,
                            int64_t *__restrict__ out);

at::Tensor hash_cuda(const at::Tensor idx);

at::Tensor kernel_hash_cuda(const at::Tensor idx,
                            const at::Tensor kernel_offset);
void kernel_hash_wrapper(int N, int K, const int *data,
                         const int *kernel_offset, int64_t *out);
void hash_wrapper(int N, const int *data, int64_t *out);
__global__ void kernel_hash_kernel(int N, int K, const int *__restrict__ data,
                                   const int *__restrict__ kernel_offset,
                                   int64_t *__restrict__ out);
