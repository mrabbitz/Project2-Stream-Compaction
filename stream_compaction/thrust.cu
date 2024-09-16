#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

struct is_zero
{
    __device__ bool operator()(const int& x) const {
        return x == 0;
    }
};

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_idata failed!");

            thrust::device_ptr<int> dev_thrust_idata(dev_idata);
            thrust::device_ptr<int> dev_thrust_odata(dev_odata);

            timer().startGpuTimer();


            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_odata to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAError("cudaFree failed!");
        }

        int compact(int n, int* odata, const int* idata)
        {
            int* dev_data;

            cudaMalloc((void**)&dev_data, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy idata to dev_data failed!");

            thrust::device_ptr<int> dev_thrust_data(dev_data);

            timer().startGpuTimer();


            thrust::device_ptr<int> new_end = thrust::remove_if(dev_thrust_data, dev_thrust_data + n, is_zero());


            timer().endGpuTimer();

            // Calculate the number of elements remaining after removal
            int compactedCount = new_end - dev_thrust_data;

            cudaMemcpy(odata, dev_data, sizeof(int) * compactedCount, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy dev_data to odata failed!");

            // Free device memory
            cudaFree(dev_data);
            checkCUDAError("cudaFree failed!");

            return compactedCount;
        }
    }
}
