#pragma once

#include <torch/torch.h>

#include <torch/extension.h>
__global__ void exclusive_scan_for_kernel_quantified(
                const int kv,
                const int *input,
                const int q,
                int *output,
                int *qoutput
);
at::Tensor exclusive_scan_quantified_wrapper(
    const int k_vol, at::Tensor neighbor_offset, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address);