#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelReductionPass(const int bufferLength, const int offset, int* data)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;

            // index out of bounds or not a rhs child node
            if ((g_index >= bufferLength) || ((g_index + 1) % (2 * offset) != 0))
            {
                return;
            }

            data[g_index] += data[g_index - offset];
        }

        __global__ void kernelDownSweepPass(const int bufferLength, const int offset, int* data)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;

            // index out of bounds or not a rhs child node
            if ((g_index >= bufferLength) || ((g_index + 1) % (2 * offset) != 0))
            {
                return;
            }

            int tmp = data[g_index];
            data[g_index] += data[g_index - offset];
            data[g_index - offset] = tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            efficientExclusivePrefixSum(n, idata, odata);
        }

        void efficientExclusivePrefixSum(const int n, const int* idata, int* odata)
        {
            int totalPasses = ilog2ceil(n);
            int bufferLength = 1 << totalPasses;

            int blocksPerGrid = (bufferLength + blockSize - 1) / blockSize;

            int* dev_data;

            cudaMalloc((void**)&dev_data, sizeof(int) * bufferLength);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_data failed!");

            if (bufferLength > n)
            {
                cudaMemset(dev_data + n, 0, sizeof(int) * (bufferLength - n));
                checkCUDAError("cudaMemset elements n to bufferLength - 1 in dev_data to 0 failed!");
            }

            timer().startGpuTimer();


            for (int offset = 1; offset < bufferLength; offset *= 2)
            {
                kernelReductionPass<<<blocksPerGrid, blockSize>>>(bufferLength, offset, dev_data);
                checkCUDAError("kernelReductionPass failed!");
            }

            cudaMemset(dev_data + bufferLength - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset last element in dev_data to 0 failed!");

            for (int offset = bufferLength / 2; offset >= 1; offset /= 2)
            {
                kernelDownSweepPass<<<blocksPerGrid, blockSize>>>(bufferLength, offset, dev_data);
                checkCUDAError("kernelDownSweepPass failed!");
            }

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");


            timer().endGpuTimer();

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
