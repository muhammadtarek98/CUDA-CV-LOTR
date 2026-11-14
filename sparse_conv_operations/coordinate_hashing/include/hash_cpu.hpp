#pragma once
#include <torch/torch.h>
#include <torch/torch.h>
#include <vector>

void cpu_hash_wrapper(int N, const int *data, int64_t *out);
void cpu_kernel_hash_wrapper(int N, int K, const int *data,
                             const int *kernel_offset, int64_t *out) ;
at::Tensor hash_cpu(const at::Tensor idx);

at::Tensor kernel_hash_cpu(const at::Tensor idx,
                           const at::Tensor kernel_offset);
