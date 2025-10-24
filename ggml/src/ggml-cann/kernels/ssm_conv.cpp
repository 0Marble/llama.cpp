
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void ggml_cann_ssm_conv_impl_kernel() {
    AscendC::printf("!!!===== RUNNING  ggml_cann_ssm_conv() DUMMY =====!!!\n");
}

void ggml_cann_ssm_conv_impl(uint32_t blockDim, void * stream) {
    ggml_cann_ssm_conv_impl_kernel<<<blockDim, nullptr, stream>>>();
}
