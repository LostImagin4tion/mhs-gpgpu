#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint merge_path_global(
    __global const uint* left,
    __global const uint* right,
    uint left_size,
    uint right_size,
    uint diagonal)
{
    uint low = diagonal > right_size ? diagonal - right_size : 0;
    uint high = min(diagonal, left_size);

    while (low < high) {
        const uint mid = low + (high - low) / 2;

        if (left[mid] <= right[diagonal - mid - 1]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

uint merge_path_local(
    __local const uint* left,
    __local const uint* right,
    uint left_size,
    uint right_size,
    uint diagonal)
{
    uint low = diagonal > right_size ? diagonal - right_size : 0;
    uint high = min(diagonal, left_size);

    while (low < high) {
        const uint mid = low + (high - low) / 2;
    
        if (left[mid] <= right[diagonal - mid - 1]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

void merge_local(
    __local const uint* left_data,
    __local const uint* right_data,
    __local uint* output,
    uint left_size,
    uint right_size,
    uint diagonal,
    uint count)
{
    uint left = merge_path_local(
        left_data,
        right_data,
        left_size,
        right_size,
        diagonal);
    uint right = diagonal - left;

    for (uint j = 0; j < count; ++j) {
        if (left < left_size && (right == right_size || left_data[left] <= right_data[right])) {
            output[j] = left_data[left++];
        } else {
            output[j] = right_data[right++];
        }
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    __local uint data[2][MERGE_SORT_BLOCK_SIZE];
    __local uint bounds[2];
    const uint lane = get_local_id(0);
    const uint block = get_group_id(0) * MERGE_SORT_BLOCK_SIZE;

    if (block >= (uint)n) {
        return;
    }

    if (sorted_k == 0) {
        for (uint j = lane; j < MERGE_SORT_BLOCK_SIZE; j += GROUP_SIZE) {
            data[0][j] = block + j < (uint)n ? input_data[block + j] : ~0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint source = 0;

        for (uint k = 1; k < MERGE_SORT_BLOCK_SIZE; k *= 2) {
            const uint count = min((uint)MERGE_SORT_VALUES_PER_THREAD, 2 * k);
    
            for (uint i = lane * count; i < MERGE_SORT_BLOCK_SIZE; i += GROUP_SIZE * count) {
                const uint start = i / (2 * k) * (2 * k);

                merge_local(
                    data[source] + start,
                    data[source] + start + k,
                    data[1 - source] + i,
                    k,
                    k,
                    i - start,
                    count);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            source = 1 - source;
        }

        for (
            uint j = lane;
            j < min((uint)MERGE_SORT_BLOCK_SIZE, (uint)n - block);
            j += GROUP_SIZE)
        {
            output_data[block + j] = data[source][j];
        }
        return;
    }

    const uint width = 2u * sorted_k;
    const uint start = block / width * width;

    const uint middle = min(start + (uint)sorted_k, (uint)n);
    const uint end = min(start + width, (uint)n);

    const uint diagonal = block - start;
    const uint count = min((uint)MERGE_SORT_BLOCK_SIZE, end - block);

    if (lane == 0) {
        bounds[0] = merge_path_global(
            input_data + start,
            input_data + middle,
            middle - start,
            end - middle,
            diagonal);

        bounds[1] = merge_path_global(
            input_data + start,
            input_data + middle,
            middle - start,
            end - middle,
            diagonal + count);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint left_size = bounds[1] - bounds[0];
    const uint right_size = count - left_size;

    for (uint j = lane; j < left_size; j += GROUP_SIZE) {
        data[0][j] = input_data[start + bounds[0] + j];
    }

    for (uint j = lane; j < right_size; j += GROUP_SIZE) {
        data[0][left_size + j] = input_data[middle + diagonal - bounds[0] + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint i = lane * MERGE_SORT_VALUES_PER_THREAD;

    if (i < count) {
        merge_local(
            data[0],
            data[0] + left_size,
            data[1] + i,
            left_size,
            right_size,
            i,
            min(
                (uint)MERGE_SORT_VALUES_PER_THREAD,
                count - i));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = lane; j < count; j += GROUP_SIZE) {
        output_data[block + j] = data[1][j];
    }
}
