#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const uint width = 2u * sorted_k;
    const uint count = min(8u, width);
    const uint i = get_global_id(0) * count;

    if (i >= (uint)n) {
        return;
    }

    const uint start = i / width * width;
    const uint middle = min(
        start + (uint)sorted_k,
        (uint)n);
    const uint end = min(
        start + width,
        (uint)n);

    const uint diagonal = i - start;
    const uint left_size = middle - start;
    const uint right_size = end - middle;

    uint low = diagonal > right_size ? diagonal - right_size : 0;
    uint high = min(diagonal, left_size);

    while (low < high) {
        const uint mid = low + (high - low) / 2;

        if (input_data[start + mid] <= input_data[middle + diagonal - mid - 1]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    uint left = start + low;
    uint right = middle + diagonal - low;

    for (uint j = i; j < min(i + count, end); ++j) {
        if (left < middle && (right == end || input_data[left] <= input_data[right])) {
            output_data[j] = input_data[left++];
        } else {
            output_data[j] = input_data[right++];
        }
    }
}
