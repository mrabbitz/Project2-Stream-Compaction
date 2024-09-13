#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void exclusivePrefixSum(const int n, const int* idata, int* odata);

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
