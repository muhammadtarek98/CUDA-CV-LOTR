#pragma once
#include <torch/torch.h>
#include <cmath>
#include <google/dense_hash_map>
#include <iostream>
#include <vector>
at::Tensor hash_query_cpu(const at::Tensor hash_query,
                          const at::Tensor hash_target,
                          const at::Tensor idx_target);
