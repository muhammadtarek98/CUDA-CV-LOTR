#include "separableconvflow_cuda_kernel.cuh"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0

int SeparableConvFlowLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  flow_output
		)
		{
	int error = 1 ;
    //int point  =0 ;printf("debug point  %d\n", point++ );

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size(0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;
    //printf("debug point  %d\n", point++ );

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;
	

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	
    int flow_output_b_stride = flow_output.stride(0);
	int flow_output_c_stride = flow_output.stride(1);
	int flow_output_h_stride = flow_output.stride(2);
	int flow_output_w_stride = flow_output.stride(3);	


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;


	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	int	nElement = 0;//UNUSED  0;//UNUSED  THCudaTensor_nElement(state, flow_output);


	error = SeparableConvFlowLayer_gpu_forward_kernel(
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,



			input1,
			input2,
			input3,
			flow_output
			
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int SeparableConvFlowLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradflow_output,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
    int flow_output_b_stride = gradflow_output.stride(0);
	int flow_output_c_stride = gradflow_output.stride(1);
	int flow_output_h_stride = gradflow_output.stride(2);
	int flow_output_w_stride = gradflow_output.stride(3);		



	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;


	int	nElement = 0;//UNUSED  0;//UNUSED  THCudaTensor_nElement(state, gradflow_output);

	error  = SeparableConvFlowLayer_gpu_backward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,

			input1,
			input2,
			input3,
			gradflow_output,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SeparableConvFlowLayer_gpu_forward", &SeparableConvFlowLayer_gpu_forward, "SeparableConvFlow forward (CUDA)");
  m.def("SeparableConvFlowLayer_gpu_backward", &SeparableConvFlowLayer_gpu_backward, "SeparableConvFlow backward (CUDA)");
}
