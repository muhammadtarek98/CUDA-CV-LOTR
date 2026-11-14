#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include "cuda_runtime.h"
#include <torch/extension.h>
#define BLOCK_SIZE (256)
#define EMPTY_CELL (0)
template <typename dtype=int>
__device__ uint64_t hash_func_64b(dtype* data);
template <typename key_type>
__device__ int hash(key_type key, int _capacity);
template <typename key_type>
__device__ int hash_murmur3(key_type key, int _capacity);

template <typename key_type=int64_t, typename val_type=int>
__global__ void insert_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, int n, int _capacity);
template <typename key_type=int64_t, typename val_type=int>
__global__ void insert_coords_kernel(key_type* table_keys, val_type* table_vals, int* coords, int n, int _capacity);
template <typename key_type=int64_t, typename val_type=int>
__global__ void lookup_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, val_type* vals, int n, int _capacity);

template <typename key_type=int64_t, typename val_type=int, bool odd>
__global__ void lookup_coords_kernel(key_type* table_keys, val_type* table_vals, int* coords, val_type* vals,
  const int* kernel_sizes, const int* strides,
  int n, int _capacity, int kernel_volume);
template <typename key_type, typename val_type>
class GPUHashTable
{
private:
    bool free_pointers;
    const int _capacity;
    const int _divisor;
    key_type* table_keys;
    val_type* table_vals;
    void insert_many_coords(int *coords, const int n);
    void lookup_many_coords(int *coords, val_type *results,
      const int* kernel_sizes, const int* tensor_strides,
      const int n, const int kernel_volume);
public:
    GPUHashTable(const int capacity);
    GPUHashTable(torch::Tensor table_keys, torch::Tensor table_vals);
    ~GPUHashTable();
    int get_divisor();
    int get_capacity();
    void insert_many(const key_type *keys, const int n);
    void lookup_many(const key_type *keys, val_type *results, const int n);
    void insert_vals(torch::Tensor keys);
    torch::Tensor lookup_vals(torch::Tensor keys);
    void insert_coords(torch::Tensor coords);
    torch::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes, at::Tensor tensor_strides, int kernel_volume);
    class device_view
    {
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
