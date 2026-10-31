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
    const uint i = get_global_id(0);

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

    const uint own_start = i < middle ? start : middle;
    const uint other_start = i < middle ? middle : start;
    
    const uint value = input_data[i];

    uint low = other_start;
    uint high = i < middle ? end : middle;

    while (low < high) {
        const uint mid = low + (high - low) / 2;

        if (input_data[mid] < value || (i >= middle && input_data[mid] == value)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    output_data[start + (i - own_start) + (low - other_start)] = value;
}
