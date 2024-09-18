#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernelReductionPass(const int reqThdsForPass, const int offset, int* data);

        __global__ void kernelDownSweepPass(const int reqThdsForPass, const int offset, int* data);

        __global__ void kernelEfficientExclusivePrefixSumByBlock(const int reqThdsPerBlock, int* data, int* blockSums);

        __global__ void kernelAddBlockSumsToBlockData(const int* blockSums, int* data);

        void scan(int n, int* odata, const int* idata, bool useSharedMemory);

        void efficientExclusivePrefixSum(const bool useGpuTimer, const int n, const int* idata, int* odata);

        void efficientExclusivePrefixSumSharedMemory(const bool useGpuTimer, const int n, const int* idata, int* odata);

        void efficientExclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int reqThdsPerBlock, const int blocksPerGrid, int* data, int* blockSums);

        int compact(int n, int* odata, const int* idata, bool useSharedMemory);
    }
}
