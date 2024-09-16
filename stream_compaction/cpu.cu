#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */

        void exclusivePrefixSum(const int n, const int* idata, int* odata)
        {
            int prefixSum = 0;

            for (int i = 0; i < n; ++i)
            {
                odata[i] = prefixSum;
                prefixSum += idata[i];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            exclusivePrefixSum(n, idata, odata);

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int compactedCount = 0;

            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[compactedCount++] = idata[i];
                }
            }

            timer().endCpuTimer();

            return compactedCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            if (n <= 0)
            {
                timer().endCpuTimer();
                return 0;
            }

            int* binaryMap = new int[n];
            int* exclusivePrefixSumResult = new int[n];

            // map the input to an array of 0s and 1s
            for (int i = 0; i < n; ++i)
            {
                binaryMap[i] = idata[i] == 0 ? 0 : 1;
            }

            // scan the array of 0s and 1s
            exclusivePrefixSum(n, binaryMap, exclusivePrefixSumResult);

            // scatter
            int compactedCount = 0;

            for (int i = 0; i < n; ++i)
            {
                if (binaryMap[i] == 1)
                {
                    odata[exclusivePrefixSumResult[i]] = idata[i];
                    ++compactedCount;
                }
            }

            timer().endCpuTimer();

            delete[] binaryMap;
            delete[] exclusivePrefixSumResult;

            return compactedCount;
        }
    }
}
