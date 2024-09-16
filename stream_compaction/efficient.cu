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

        __global__ void kernelEfficientExclusivePrefixSumByBlock(const int n, int* data, int* sums)
        {
            extern __shared__ int shared[];

            int tx = threadIdx.x;
            int g_index = blockIdx.x * blockDim.x + tx;

            // Load input into shared memory
            if (g_index < n)
            {
                shared[tx] = data[g_index];
            }
            else
            {
                shared[tx] = 0;
            }
            __syncthreads();

            // Upsweep (Reduction) phase
            for (int offset = 1; offset < blockSize; offset *= 2)
            {
                if ((tx + 1) % (2 * offset) == 0)
                {
                    shared[tx] += shared[tx - offset];
                }
                __syncthreads();
            }

            // Clear the last element
            if (tx == blockSize - 1)
            {
                sums[blockIdx.x] = shared[tx];
                shared[tx] = 0;
            }
            __syncthreads();

            // Downsweep phase
            for (int offset = blockSize / 2; offset > 0; offset /= 2)
            {
                if ((tx + 1) % (2 * offset) == 0)
                {
                    int temp = shared[tx];
                    shared[tx] += shared[tx - offset];
                    shared[tx - offset] = temp;
                }
                __syncthreads();
            }

            // Write the results back to global memory
            if (g_index < n)
            {
                data[g_index] = shared[tx];
            }
        }

        __global__ void kernelAddBlockSumsToBlockData(const int n, const int* idataBlockSums, int* data)
        {
            int g_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (g_index >= n)
            {
                return;
            }

            data[g_index] += idataBlockSums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool useSharedMemory)
        {
            if (useSharedMemory)
            {
                efficientExclusivePrefixSumSharedMemory(true, n, idata, odata);
            }
            else
            {
                efficientExclusivePrefixSum(true, n, idata, odata);
            }
        }

        void efficientExclusivePrefixSum(const bool useGpuTimer, const int n, const int* idata, int* odata)
        {
            int totalPasses = ilog2ceil(n);
            int bufferLength = 1 << totalPasses;

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

            int blocksPerGrid = (bufferLength + blockSize - 1) / blockSize;

            if (useGpuTimer) timer().startGpuTimer();


            for (int offset = 1; offset < bufferLength; offset *= 2)
            {
                kernelReductionPass<<<blocksPerGrid, blockSize>>>(bufferLength, offset, dev_data);
                checkCUDAError("kernelReductionPass failed!");
            }

            cudaMemset(dev_data + bufferLength - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset last element in dev_data to 0 failed!");

            for (int offset = bufferLength / 2; offset > 0; offset /= 2)
            {
                kernelDownSweepPass<<<blocksPerGrid, blockSize>>>(bufferLength, offset, dev_data);
                checkCUDAError("kernelDownSweepPass failed!");
            }


            if (useGpuTimer) timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");

            cudaFree(dev_data);
        }

        void efficientExclusivePrefixSumSharedMemory(const bool useGpuTimer, const int n, const int* idata, int* odata)
        {
            int blocksPerGrid = (n + blockSize - 1) / blockSize;

            int* dev_data;
            int* dev_sums;

            cudaMalloc((void**)&dev_data, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMalloc((void**)&dev_sums, sizeof(int) * blocksPerGrid);
            checkCUDAError("cudaMalloc dev_sums failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_data failed!");

            int sharedMemoryBytes = blockSize * sizeof(int);

            if (useGpuTimer) timer().startGpuTimer();


            efficientExclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes, n, blocksPerGrid, dev_data, dev_sums);


            if (useGpuTimer) timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");

            cudaFree(dev_data);
            cudaFree(dev_sums);
        }

        // iterative approach is possible
        // for the sake of submitting this assignement on time, this will have to be explored at a later time
        void efficientExclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int n, const int numBlocks, int* data, int* sums)
        {
            kernelEfficientExclusivePrefixSumByBlock<<<numBlocks, blockSize, sharedMemoryBytes>>>(n, data, sums);
            checkCUDAError("kernelEfficientExclusivePrefixSumByBlock failed!");

            if (numBlocks > 1)
            {
                int numBlocksForBlockSums = (numBlocks + blockSize - 1) / blockSize;

                int* dev_sums;

                cudaMalloc((void**)&dev_sums, sizeof(int) * numBlocksForBlockSums);
                checkCUDAError("cudaMalloc dev_sums failed!");

                efficientExclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes, numBlocks, numBlocksForBlockSums, sums, dev_sums);

                kernelAddBlockSumsToBlockData<<<numBlocks, blockSize>>>(n, sums, data);
                checkCUDAError("kernelAddBlockSumsToBlockData failed!");

                cudaFree(dev_sums);
            }
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
        int compact(int n, int* odata, const int* idata, bool useSharedMemory)
        {
            int* dev_idata;
            int* dev_odata;
            int* dev_binaryMap;
            int* dev_exclusivePrefixSumResult;

            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_binaryMap, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_binaryMap failed!");

            cudaMalloc((void**)&dev_exclusivePrefixSumResult, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_exclusivePrefixSumResult failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_idata failed!");

            int blocksPerGrid = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();


            Common::kernMapToBoolean<<<blocksPerGrid, blockSize>>>(n, dev_binaryMap, dev_idata);

            if (useSharedMemory)
            {
                efficientExclusivePrefixSumSharedMemory(false, n, dev_binaryMap, dev_exclusivePrefixSumResult);
            }
            else
            {
                efficientExclusivePrefixSum(false, n, dev_binaryMap, dev_exclusivePrefixSumResult);
            }

            Common::kernScatter<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_binaryMap, dev_exclusivePrefixSumResult);


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_bufferB to odata failed!");

            int compactedCount;
            cudaMemcpy(&compactedCount, dev_exclusivePrefixSumResult + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_exclusivePrefixSumResult to compactedCount failed!");

            if (idata[n - 1] != 0)
            {
                ++compactedCount;
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_binaryMap);
            cudaFree(dev_exclusivePrefixSumResult);

            return compactedCount;
        }
    }
}
