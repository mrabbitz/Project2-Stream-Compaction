#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernelReductionPass(const int bufferLength, const int offset, int* data);

        __global__ void kernelDownSweepPass(const int bufferLength, const int offset, int* data);

        void scan(int n, int* odata, const int* idata);

        void efficientExclusivePrefixSum(const int n, const int* idata, int* odata);

        int compact(int n, int* odata, const int* idata);
    }
}
