#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernelReductionPass(const int bufferLength, const int offset, int* data);

        __global__ void kernelDownSweepPass(const int bufferLength, const int offset, int* data);

        __global__ void kernelEfficientExclusivePrefixSumByBlock(const int n, int* data, int* sums);

        __global__ void kernelAddBlockSumsToBlockData(const int n, const int* idataBlockSums, int* data);

        void scan(int n, int* odata, const int* idata, bool useSharedMemory);

        void efficientExclusivePrefixSum(const bool useGpuTimer, const int n, const int* idata, int* odata);

        void efficientExclusivePrefixSumSharedMemory(const bool useGpuTimer, const int n, const int* idata, int* odata);

        void efficientExclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int n, const int numBlocks, int* data, int* sums);

        int compact(int n, int* odata, const int* idata, bool useSharedMemory);
    }
}
