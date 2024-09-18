#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
 ((n) >> (LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelReductionPass(const int reqThdsForPass, const int offset, int* data)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (g_index >= reqThdsForPass)
            {
                return;
            }

            int left_node_g_index = offset * (2 * g_index + 1) - 1;
            int right_node_g_index = offset * (2 * g_index + 2) - 1;

            data[right_node_g_index] += data[left_node_g_index];
        }

        __global__ void kernelDownSweepPass(const int reqThdsForPass, const int offset, int* data)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (g_index >= reqThdsForPass)
            {
                return;
            }

            int left_node_g_index = offset * (2 * g_index + 1) - 1;
            int right_node_g_index = offset * (2 * g_index + 2) - 1;

            int temp = data[left_node_g_index];
            data[left_node_g_index] = data[right_node_g_index];
            data[right_node_g_index] += temp;
        }

        __global__ void kernelEfficientExclusivePrefixSumByBlock(const int reqThdsPerBlock, int* data, int* blockSums)
        {
            // allocated on invocation
            extern __shared__ int shared[];

            int tx = threadIdx.x;

            if (tx >= reqThdsPerBlock)
            {
                return;
            }

            int g_index = 2 * (blockIdx.x * blockDim.x) + tx;

            int ai = tx;
            int bi = tx + reqThdsPerBlock;
            int g_ai = g_index;
            int g_bi = g_index + reqThdsPerBlock;

            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            // Load input into shared memory
            shared[ai + bankOffsetA] = data[g_ai];
            shared[bi + bankOffsetB] = data[g_bi];

            int offset = 1;

            // Upsweep (Reduction) phase
            for (int reqThdsForPass = reqThdsPerBlock; reqThdsForPass > 0; reqThdsForPass >>= 1)
            {
                __syncthreads();

                if (tx < reqThdsForPass)
                {
                    int left_node_shrd_index = offset * (2 * tx + 1) - 1;
                    int right_node_shrd_index = offset * (2 * tx + 2) - 1;
                    left_node_shrd_index += CONFLICT_FREE_OFFSET(left_node_shrd_index);
                    right_node_shrd_index += CONFLICT_FREE_OFFSET(right_node_shrd_index);

                    shared[right_node_shrd_index] += shared[left_node_shrd_index];
                }

                offset *= 2;
            }

            // Write the block sum and then clear it for the Downsweep
            if (tx == 0)
            {
                int n_minus_1 = 2 * reqThdsPerBlock - 1;
                n_minus_1 += CONFLICT_FREE_OFFSET(n_minus_1);

                blockSums[blockIdx.x] = shared[n_minus_1];
                shared[n_minus_1] = 0;
            }

            // Downsweep phase
            for (int reqThdsForPass = 1; reqThdsForPass <= reqThdsPerBlock; reqThdsForPass *= 2)
            {
                offset >>= 1;
                __syncthreads();

                if (tx < reqThdsForPass)
                {
                    int left_node_shrd_index = offset * (2 * tx + 1) - 1;
                    int right_node_shrd_index = offset * (2 * tx + 2) - 1;
                    left_node_shrd_index += CONFLICT_FREE_OFFSET(left_node_shrd_index);
                    right_node_shrd_index += CONFLICT_FREE_OFFSET(right_node_shrd_index);

                    int temp = shared[left_node_shrd_index];
                    shared[left_node_shrd_index] = shared[right_node_shrd_index];
                    shared[right_node_shrd_index] += temp;
                }
            }

            __syncthreads();

            // Write the results back to global memory
            data[g_ai] = shared[ai + bankOffsetA];
            data[g_bi] = shared[bi + bankOffsetB];
        }

        __global__ void kernelAddBlockSumsToBlockData(const int* blockSums, int* data)
        {
            int g_index = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);

            //     *****     blockSize must be a power of 2     *****     //
            // no reason to check bounds because this kernel is only called if we recursively scan,
            // meaning the buffer length (of int* data) is going to be (2 * (blockSize)) * (2 to the power of n),
            // where n is the depth of recursion. To reiterate the first line, n is always >= 1

            data[g_index] += blockSums[blockIdx.x];
            data[g_index + 1] += blockSums[blockIdx.x];
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

            int blocksPerGrid = 0;
            int offset = 1;

            if (useGpuTimer) timer().startGpuTimer();


            for (int reqThdsForPass = bufferLength >> 1; reqThdsForPass > 0; reqThdsForPass >>= 1)
            {
                blocksPerGrid = (reqThdsForPass + blockSize - 1) / blockSize;
                kernelReductionPass<<<blocksPerGrid, blockSize>>>(reqThdsForPass, offset, dev_data);
                checkCUDAError("kernelReductionPass failed!");

                offset *= 2;
            }

            cudaMemset(dev_data + bufferLength - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset last element in dev_data to 0 failed!");

            for (int reqThdsForPass = 1; reqThdsForPass < bufferLength; reqThdsForPass *= 2)
            {
                offset >>= 1;

                blocksPerGrid = (reqThdsForPass + blockSize - 1) / blockSize;
                kernelDownSweepPass<<<blocksPerGrid, blockSize>>>(reqThdsForPass, offset, dev_data);
                checkCUDAError("kernelDownSweepPass failed!");
            }


            if (useGpuTimer) timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");

            cudaFree(dev_data);
            checkCUDAError("cudaFree failed!");
        }

        void efficientExclusivePrefixSumSharedMemory(const bool useGpuTimer, const int n, const int* idata, int* odata)
        {
            int totalPasses = ilog2ceil(n);
            int bufferLength = 1 << totalPasses;
            int reqThdsTotal = bufferLength >> 1;
            int blocksPerGrid = (reqThdsTotal + blockSize - 1) / blockSize;

            int* dev_data;
            int* dev_sums;

            cudaMalloc((void**)&dev_data, sizeof(int) * bufferLength);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMalloc((void**)&dev_sums, sizeof(int) * blocksPerGrid);
            checkCUDAError("cudaMalloc dev_sums failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_data failed!");

            if (bufferLength > n)
            {
                cudaMemset(dev_data + n, 0, sizeof(int) * (bufferLength - n));
                checkCUDAError("cudaMemset elements n to bufferLength - 1 in dev_data to 0 failed!");
            }

            int reqThdsPerBlock = std::min(reqThdsTotal, blockSize);
            int procElemsPerBlock = 2 * reqThdsPerBlock;
            int sharedMemoryBytes = sizeof(int) * (procElemsPerBlock + (procElemsPerBlock / NUM_BANKS) + (procElemsPerBlock / (NUM_BANKS * NUM_BANKS)));

            if (useGpuTimer) timer().startGpuTimer();

            
            efficientExclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes, reqThdsPerBlock, blocksPerGrid, dev_data, dev_sums);


            if (useGpuTimer) timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");

            cudaFree(dev_data);
            cudaFree(dev_sums);
            checkCUDAError("cudaFree failed!");
        }

        // iterative approach is possible
        // for the sake of submitting this assignement on time, this will have to be explored at a later time
        void efficientExclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int reqThdsPerBlock, const int blocksPerGrid, int* data, int* blockSums)
        {
            kernelEfficientExclusivePrefixSumByBlock<<<blocksPerGrid, blockSize, sharedMemoryBytes>>>(reqThdsPerBlock, data, blockSums);
            checkCUDAError("kernelEfficientExclusivePrefixSumByBlock failed!");

            if (blocksPerGrid > 1)
            {
                int n_blockSums = blocksPerGrid;

                int totalPasses_blockSums = ilog2ceil(n_blockSums);
                int bufferLength_blockSums = 1 << totalPasses_blockSums;
                int reqThdsTotal_blockSums = bufferLength_blockSums >> 1;
                int blocksPerGrid_blockSums = (reqThdsTotal_blockSums + blockSize - 1) / blockSize;

                int* dev_sums;
                int* dev_new_sums;

                cudaMalloc((void**)&dev_sums, sizeof(int) * bufferLength_blockSums);
                checkCUDAError("cudaMalloc dev_sums failed!");

                cudaMalloc((void**)&dev_new_sums, sizeof(int) * blocksPerGrid_blockSums);
                checkCUDAError("cudaMalloc dev_new_sums failed!");

                cudaMemcpy(dev_sums, blockSums, sizeof(int) * n_blockSums, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy blockSums to dev_sums failed!");

                if (bufferLength_blockSums > n_blockSums)
                {
                    cudaMemset(dev_sums + n_blockSums, 0, sizeof(int) * (bufferLength_blockSums - n_blockSums));
                    checkCUDAError("cudaMemset elements n_blockSums to bufferLength_blockSums - 1 in dev_sums to 0 failed!");
                }

                int reqThdsPerBlock_blockSums = std::min(reqThdsTotal_blockSums, blockSize);
                int procElemsPerBlock_blockSums = 2 * reqThdsPerBlock_blockSums;
                int sharedMemoryBytes_blockSums = sizeof(int) * (procElemsPerBlock_blockSums + (procElemsPerBlock_blockSums / NUM_BANKS) + (procElemsPerBlock_blockSums / (NUM_BANKS * NUM_BANKS)));

                efficientExclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes_blockSums, reqThdsPerBlock_blockSums, blocksPerGrid_blockSums, dev_sums, dev_new_sums);

                kernelAddBlockSumsToBlockData<<<n_blockSums, blockSize>>>(dev_sums, data);
                checkCUDAError("kernelAddBlockSumsToBlockData failed!");

                cudaFree(dev_sums);
                cudaFree(dev_new_sums);
                checkCUDAError("cudaFree failed!");
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
            checkCUDAError("kernMapToBoolean failed!");

            if (useSharedMemory)
            {
                efficientExclusivePrefixSumSharedMemory(false, n, dev_binaryMap, dev_exclusivePrefixSumResult);
            }
            else
            {
                efficientExclusivePrefixSum(false, n, dev_binaryMap, dev_exclusivePrefixSumResult);
            }

            Common::kernScatter<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_binaryMap, dev_exclusivePrefixSumResult);
            checkCUDAError("kernScatter failed!");


            timer().endGpuTimer();

            int compactedCount;
            cudaMemcpy(&compactedCount, dev_exclusivePrefixSumResult + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_exclusivePrefixSumResult last element to compactedCount failed!");

            if (idata[n - 1] != 0)
            {
                ++compactedCount;
            }

            cudaMemcpy(odata, dev_odata, sizeof(int) * compactedCount, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_odata to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_binaryMap);
            cudaFree(dev_exclusivePrefixSumResult);
            checkCUDAError("cudaFree failed!");

            return compactedCount;
        }
    }
}
