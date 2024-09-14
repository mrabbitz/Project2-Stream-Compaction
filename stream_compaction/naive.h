#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernelInclusiveToExclusivePrefixSum(const int n, const int* idata, int* odata);

        __global__ void kernelNaiveInclusivePrefixSumPass(const int n, const int offset, const int* idata, int* odata);

        __global__ void kernelNaiveInclusivePrefixSumByBlock(const int n, const int* idata, int* odata);

        __global__ void kernelNaiveExclusivePrefixSumByBlock(const int n, const int* idata, int* odata);

        __global__ void kernelExtractBlockSums(const int n, const int numBlocks, const int* idata, int* odata);

        __global__ void kernelAddBlockIncrements(const int n, const int* idataBlockSums, const int* idata, int* odata);

        void scan(int n, int* odata, const int* idata);

        void naiveExclusivePrefixSum(const int n, const int* idata, int* odata);

        void naiveExclusivePrefixSumSharedMemory(const int n, const int* idata, int* odata);

        void naiveInclusivePrefixSumAnyNumberOfBlocks(const int sharedMemoryBytes, const int n, const int numBlocks, int* idata, int* odata);
    }
}
