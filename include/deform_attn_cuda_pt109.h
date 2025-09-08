#pragma once
#include <torch/extension.h>
#include <ATen/DeviceGuard.h>
#include <bits/stdc++.h>
void deformable_im2col(
const at::Tensor &data_im, const at::Tensor &data_offset,const int &channels, const int &height,
const int &width,const int &ksize_h, const int &ksize_w, const int &pad_h,const int &pad_w,
const int &stride_h, const int &stride_w,const int &dilation_h, const int &dilation_w,
const int &parallel_imgs, const int &deform_group,at::Tensor data_col);

void deformable_col2im(
const at::Tensor &data_col, const at::Tensor &data_offset,const int &channels, const int &height,
const int &width,const int &ksize_h, const int &ksize_w, const int &pad_h,const int &pad_w,
const int &stride_h, const int &stride_w,const int &dilation_h, const int& dilation_w,
const int &parallel_imgs, const int &deform_group,at::Tensor grad_im);

void deformable_col2im_coord(
const at::Tensor &data_col, const at::Tensor &data_im,
const at::Tensor &data_offset, const int &channels, const int &height,
const int &width, const int &ksize_h, const int &ksize_w, const int &pad_h,
const int &pad_w, const int &stride_h, const int &stride_w,
const int &dilation_h, const int &dilation_w, const int &parallel_imgs,
const int &deform_group, at::Tensor grad_offset);

void modulated_deformable_im2col_cuda(
const at::Tensor &data_im, const at::Tensor &data_offset,
const at::Tensor &data_mask, const int &batch_size, const int &channels,
const int &height_im, const int &width_im, const int &height_col,
const int &width_col, const int &kernel_h, const int &kenerl_w,
const int &pad_h, const int &pad_w, const int &stride_h, const int &stride_w,
const int &dilation_h, const int &dilation_w, const int &deform_group,
at::Tensor data_col);

void modulated_deformable_col2im_cuda(
const at::Tensor &data_col, const at::Tensor &data_offset,
const at::Tensor &data_mask, const int &batch_size, const int &channels,
const int &height_im, const int &width_im, const int &height_col,
const int &width_col, const int &kernel_h, const int &kenerl_w,
const int &pad_h, const int &pad_w, const int &stride_h, const int &stride_w,
const int &dilation_h, const int &dilation_w, const int &deform_group,
at::Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
const at::Tensor &data_col, const at::Tensor &data_im,
const at::Tensor &data_offset, const at::Tensor &data_mask,
const int &batch_size, const int &channels, const int &height_im,
const int &width_im, const int &height_col, const int &width_col,
const int &kernel_h, const int &kenerl_w, const int &pad_h, const int &pad_w,
const int &stride_h, const int &stride_w, const int &dilation_h,
const int &dilation_w, const int &deform_group, at::Tensor grad_offset, at::Tensor grad_mask);

void deform_attn_cuda_forward(
at::Tensor q, at::Tensor kv, at::Tensor offset, at::Tensor &output,
at::Tensor columns, at::Tensor &attns, at::Tensor mask_ones, int kernel_h, int kernel_w,
const int &stride_h,const int &stride_w,const int &pad_h, const int &pad_w, const int &dilation_h,
const int &dilation_w, const int &attn_head, const int &deform_group, const int &clip_size);
void deform_attn_cuda_backward(
at::Tensor q, at::Tensor kv, at::Tensor offset,
at::Tensor columns, at::Tensor &attns, at::Tensor mask_ones, at::Tensor grad_attns,
at::Tensor grad_mask_ones, at::Tensor &grad_q, at::Tensor grad_kv,at::Tensor &grad_offset,
at::Tensor grad_output,int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
int pad_w, int dilation_h, int dilation_w, int attn_head, int deform_group, int clip_size);
