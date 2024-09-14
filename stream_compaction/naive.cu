#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define SHARED_MEMORY 1

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelInclusiveToExclusivePrefixSum(const int n, const int* idata, int* odata)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (g_index >= n)
            {
                return;
            }

            if (g_index > 0)
            {
                odata[g_index] = idata[g_index - 1];
            }
            else if (g_index == 0)
            {
                odata[g_index] = 0;
            }
        }

        __global__ void kernelNaiveInclusivePrefixSumPass(const int n, const int offset, const int* idata, int* odata)
        {
            int g_index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (g_index >= n)
            {
                return;
            }

            if (g_index >= offset)
            {
                odata[g_index] = idata[g_index - offset] + idata[g_index];
            }
            else
            {
                odata[g_index] = idata[g_index];
            }
        }

        __global__ void kernelNaiveInclusivePrefixSumByBlock(const int n, const int* idata, int* odata)
        {
            int g_index = blockIdx.x * blockDim.x + threadIdx.x;

            if (g_index >= n)
            {
                return;
            }

            // allocated on invocation
            extern __shared__ int doubleBuffer[];

            int tx = threadIdx.x;

            int pout = 0, pin = 1;

            // Load input into shared memory.
            doubleBuffer[pout * blockSize + tx] = idata[g_index];
            doubleBuffer[pin * blockSize + tx] = doubleBuffer[pout * blockSize + tx];
            __syncthreads();

            for (int offset = 1; offset < blockSize; offset *= 2)
            {
                // swap double buffer indices
                pout = 1 - pout;
                pin = 1 - pout;

                if (tx >= offset)
                {
                    doubleBuffer[pout * blockSize + tx] = doubleBuffer[pin * blockSize + tx - offset] + doubleBuffer[pin * blockSize + tx];
                }
                else
                {
                    doubleBuffer[pout * blockSize + tx] = doubleBuffer[pin * blockSize + tx];
                }
                __syncthreads();
            }

            // write output
            odata[g_index] = doubleBuffer[pout * blockSize + tx];
        }

        __global__ void kernelNaiveExclusivePrefixSumByBlock(const int n, const int* idata, int* odata)
        {
            int g_index = blockIdx.x * blockDim.x + threadIdx.x;

            if (g_index >= n)
            {
                return;
            }

            // allocated on invocation
            extern __shared__ int doubleBuffer[];

            int tx = threadIdx.x;

            int pout = 0, pin = 1;

            // Load input into shared memory.
            // This is exclusive scan, so shift right by one
            // and set first element to 0
            doubleBuffer[pout * blockSize + tx] = (tx > 0) ? idata[g_index - 1] : 0;
            doubleBuffer[pin * blockSize + tx] = doubleBuffer[pout * blockSize + tx];
            __syncthreads();

            for (int offset = 1; offset < blockSize; offset *= 2)
            {
                // swap double buffer indices
                pout = 1 - pout;
                pin = 1 - pout;

                if (tx >= offset)
                {
                    doubleBuffer[pout * blockSize + tx] = doubleBuffer[pin * blockSize + tx - offset] + doubleBuffer[pin * blockSize + tx];
                }
                else
                {
                    doubleBuffer[pout * blockSize + tx] = doubleBuffer[pin * blockSize + tx];
                }
                __syncthreads();
            }

            // write output
            odata[g_index] = doubleBuffer[pout * blockSize + tx];
        }

        __global__ void kernelExtractBlockSums(const int n, const int numBlocks, const int* idata, int* odata)
        {
            int g_index = blockIdx.x * blockDim.x + threadIdx.x;

            if (g_index >= numBlocks)
            {
                return;
            }

            odata[g_index] = g_index == numBlocks - 1 ? idata[n - 1] : idata[(g_index * blockSize) + blockSize - 1];
        }

        __global__ void kernelAddBlockIncrements(const int n, const int* idataBlockSums, const int* idata, int* odata)
        {
            int g_index = blockIdx.x * blockDim.x + threadIdx.x;

            if (g_index >= n)
            {
                return;
            }

            odata[g_index] = idata[g_index] + idataBlockSums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata)
        {
            #if SHARED_MEMORY
            naiveExclusivePrefixSumSharedMemory(n, idata, odata);
            #else
            naiveExclusivePrefixSum(n, idata, odata);
            #endif
        }

        void naiveExclusivePrefixSum(const int n, const int* idata, int* odata)
        {
            int* dev_bufferA;
            int* dev_bufferB;

            cudaMalloc((void**)&dev_bufferA, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bufferA failed!");

            cudaMalloc((void**)&dev_bufferB, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bufferB failed!");

            cudaMemcpy(dev_bufferA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_bufferA failed!");

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            
            for (int offset = 1; offset < n; offset *= 2)
            {
                kernelNaiveInclusivePrefixSumPass<<<blocksPerGrid, blockSize>>>(n, offset, dev_bufferA, dev_bufferB);
                checkCUDAError("kernelNaiveInclusivePrefixSumPass failed!");

                // set the input of the next iteration to the output of this iteration
                std::swap(dev_bufferA, dev_bufferB);
            }

            kernelInclusiveToExclusivePrefixSum<<<blocksPerGrid, blockSize>>>(n, dev_bufferA, dev_bufferB);
            checkCUDAError("kernelInclusiveToExclusivePrefixSum failed!");

            cudaMemcpy(odata, dev_bufferB, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_bufferB to odata failed!");


            timer().endGpuTimer();

            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
        }

        void naiveExclusivePrefixSumSharedMemory(const int n, const int* idata, int* odata)
        {
            int* dev_bufferA;
            int* dev_bufferB;

            cudaMalloc((void**)&dev_bufferA, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bufferA failed!");

            cudaMalloc((void**)&dev_bufferB, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bufferB failed!");

            cudaMemcpy(dev_bufferA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_bufferA failed!");

            const int sharedMemoryBytes = 2 * blockSize * sizeof(int);

            int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();


            naiveInclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes, n, numBlocks, dev_bufferA, dev_bufferB);

            kernelInclusiveToExclusivePrefixSum<<<(n + blockSize - 1) / blockSize, blockSize>>>(n, dev_bufferB, dev_bufferA);
            checkCUDAError("kernelInclusiveToExclusivePrefixSum failed!");

            cudaMemcpy(odata, dev_bufferA, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_bufferA to odata failed!");


            timer().endGpuTimer();

            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
        }

        // iterative approach is possible if the blockSums buffers are allocated carefully ahead of time, combined with clever indexing of them at each iteration
        // for the sake of submitting this assignement on time, this will have to be explored at a later time
        void naiveInclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int n, const int numBlocks, int* idata, int* odata)
        {
            kernelNaiveInclusivePrefixSumByBlock<<<numBlocks, blockSize, sharedMemoryBytes>>>(n, idata, odata);
            checkCUDAError("kernelNaiveInclusivePrefixSumByBlock failed!");

            if (numBlocks > 1)
            {
                int* dev_bufferBlockSumsA;
                int* dev_bufferBlockSumsB;

                cudaMalloc((void**)&dev_bufferBlockSumsA, sizeof(int) * numBlocks);
                checkCUDAError("cudaMalloc dev_bufferBlockSumsA failed!");

                cudaMalloc((void**)&dev_bufferBlockSumsB, sizeof(int) * numBlocks);
                checkCUDAError("cudaMalloc dev_bufferBlockSumsB failed!");

                int numBlocksForBlockSums = (numBlocks + blockSize - 1) / blockSize;
                kernelExtractBlockSums<<<numBlocksForBlockSums, blockSize>>>(n, numBlocks, odata, dev_bufferBlockSumsA);
                checkCUDAError("kernelExtractBlockSums failed!");

                if (numBlocksForBlockSums > 1)
                {
                    naiveInclusivePrefixSumAnyNumberOfBlocks(sharedMemoryBytes, numBlocks, numBlocksForBlockSums, dev_bufferBlockSumsA, dev_bufferBlockSumsB);

                    kernelInclusiveToExclusivePrefixSum<<<numBlocksForBlockSums, blockSize>>>(numBlocks, dev_bufferBlockSumsB, dev_bufferBlockSumsA);
                    checkCUDAError("kernelInclusiveToExclusivePrefixSum failed!");

                    kernelAddBlockIncrements<<<numBlocks, blockSize>>>(n, dev_bufferBlockSumsA, odata, idata);
                    checkCUDAError("kernelAddBlockIncrements failed!");
                }
                else
                {
                    kernelNaiveExclusivePrefixSumByBlock<<<numBlocksForBlockSums, blockSize, sharedMemoryBytes>>>(numBlocks, dev_bufferBlockSumsA, dev_bufferBlockSumsB);
                    checkCUDAError("kernelNaiveExclusivePrefixSumByBlock failed!");

                    kernelAddBlockIncrements<<<numBlocks, blockSize>>>(n, dev_bufferBlockSumsB, odata, idata);
                    checkCUDAError("kernelAddBlockIncrements failed!");
                }

                cudaMemcpy(odata, idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy idata to odata failed!");

                cudaFree(dev_bufferBlockSumsA);
                cudaFree(dev_bufferBlockSumsB);
            }
        }
    }
}