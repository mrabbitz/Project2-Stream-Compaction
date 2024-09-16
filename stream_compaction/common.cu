#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {

            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (g_index >= n)
            {
                return;
            }

            bools[g_index] = idata[g_index] == 0 ? 0 : 1;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {

            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;

            // index out of bounds or not a "hit"
            if ((g_index >= n) || (bools[g_index] == 0))
            {
                return;
            }

            odata[indices[g_index]] = idata[g_index];
        }

    }
}
