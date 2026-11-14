#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include "cuda_runtime.h"
#include <torch/extension.h>
/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)

__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val) {
  int32_t* base_address = (int32_t*)((char*)address - (reinterpret_cast<size_t>(address) & 3));
  int32_t int_val = static_cast<int32_t>(val) << ((reinterpret_cast<size_t>(address) & 3) * 8);
  int32_t int_comp = static_cast<int32_t>(compare) << ((reinterpret_cast<size_t>(address) & 3) * 8);
  return static_cast<int8_t>(atomicCAS(base_address, int_comp, int_val));
}

// TODO: can we do this more efficiently?
__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val) {
  int32_t* base_address = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
  int32_t int_val = static_cast<int32_t>(val) << ((reinterpret_cast<size_t>(address) & 2) * 8);
  int32_t int_comp = static_cast<int32_t>(compare) << ((reinterpret_cast<size_t>(address) & 2) * 8);
  return static_cast<int16_t>(atomicCAS(base_address, int_comp, int_val));
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(compare),
                            static_cast<unsigned long long>(val));
}

template <typename dtype=int>
__device__ uint64_t hash_func_64b(dtype* data);

template <typename key_type>
__device__ int hash(key_type key, int _capacity);

template <typename key_type>
__device__ int hash_murmur3(key_type key, int _capacity);
template <typename key_type, typename val_type>
class GPUHashTable {
 private:
  bool free_pointers;
  const int _capacity;
  const int _divisor;
  key_type* table_keys;
  val_type* table_vals;
  void insert_many_coords(int *coords, const int n);
  void lookup_many_coords(int *coords, val_type *results,const int* kernel_sizes, const int* tensor_strides,const int n, const int kernel_volume);
 public:
  GPUHashTable(const int capacity);
  GPUHashTable(torch::Tensor table_keys, torch::Tensor table_vals);
  ~GPUHashTable() ;
  void insert_many(const key_type *keys, const int n);
  void lookup_many(const key_type *keys, val_type *results, const int n);
  void insert_vals(torch::Tensor keys);
  torch::Tensor lookup_vals(torch::Tensor keys);
  void insert_coords(torch::Tensor coords);
  torch::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes, at::Tensor tensor_strides, int kernel_volume);
  int get_divisor(){return _divisor;}
  int get_capacity(){return _capacity;}
  class device_view{
    private:
      int _capacity;
      key_type* _table_keys;
      val_type* _table_vals;
    public:
      __host__ __device__ device_view(int capacity, key_type* table_keys, val_type* table_vals);
      __device__ val_type lookup(const key_type key);
      __device__ void insert(const key_type key, const val_type val);
  };
  __host__ __device__ device_view get_device_view();
};

using hashtable = GPUHashTable<int64_t, int>;
using hashtable32 = GPUHashTable<int, int>;

// Insert into hashmap
template <typename key_type=int64_t, typename val_type=int>
__global__ void insert_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, int n, int _capacity);


template <typename key_type=int64_t, typename val_type=int>
__global__ void insert_coords_kernel(key_type* table_keys, val_type* table_vals, int* coords, int n, int _capacity);

// lookup from hashmap
template <typename key_type=int64_t, typename val_type=int>
__global__ void lookup_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, val_type* vals, int n, int _capacity);


template <typename key_type=int64_t, typename val_type=int, bool odd>
__global__ void lookup_coords_kernel(
  key_type* table_keys, val_type* table_vals, int* coords, val_type* vals, 
  const int* kernel_sizes, const int* strides, 
  int n, int _capacity, int kernel_volume);