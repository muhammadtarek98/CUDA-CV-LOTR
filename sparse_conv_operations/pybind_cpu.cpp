#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "convolution_gather_scatter_cpu.hpp"
#include "devoxelize_cpu.hpp"
#include "hash_cpu.hpp"
#include "count_cpu.hpp"
#include "query_cpu.hpp"
#include "voxelize_cpu.hpp"

void init_cpu_ops(py::module_ &m) {
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("hash_cpu", &hash_cpu);
  m.def("kernel_hash_cpu", &kernel_hash_cpu);
  m.def("hash_query_cpu", &hash_query_cpu);
  m.def("count_cpu", &count_cpu);
}
