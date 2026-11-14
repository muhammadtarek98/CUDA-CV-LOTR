#pragma once

template <int bytes>
struct global_load;

template <>
struct global_load<16>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard);
};
template <>
struct global_load<8>;
template <>
struct global_load<4>;
template <>
struct global_load<2>;