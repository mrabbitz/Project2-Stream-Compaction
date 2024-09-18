CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Michael Rabbitz
  * [LinkedIn](https://www.linkedin.com/in/mike-rabbitz)
* Tested on: Windows 10, i7-9750H @ 2.60GHz 32GB, RTX 2060 6GB (Personal)

## Introduction

This project focuses on implementing various Stream Compaction algorithms, including those utilizing the Scan algorithm, to emphasize the importance of designing GPU hardware-optimized algorithms that leverage parallel computation for superior performance compared to CPU implementations.

### Stream Compaction
Stream compaction involves filtering an input data set to produce a new collection that contains only the elements meeting a specified condition, while preserving their original order. This process reduces the size of the data set by removing unwanted elements, which is crucial for optimizing performance and memory usage in applications like path tracing, collision detection, etc.

![Stream Compaction](img/stream_compaction_visual.PNG)

### Scan
The Scan algorithm, also known as the all-prefix-sums operation, computes prefix sums for an array of data. In an exclusive scan, each element in the result is the sum of all preceding elements in the input array. Conversely, an inclusive scan includes the current element in the sum. Scan algorithms are fundamental in parallel computing for tasks like sorting, stream compaction, and building data structures, as they transform inherently sequential computations into parallel ones.

![Scan](img/scan_visual.PNG)

## Implementation Details

**n represents the number of elements in the array**

**All implementations support arrays of arbitrary n - small, large, powers of two, not powers of two**

### Scan


1. **CPU:**  O(n) addition operations - sequential loop over array elements, accumulating a sum at each iteration
2. **GPU Naive Algorithm:**  O(n * log<sub>2</sub>(n)) addition operations - over log<sub>2</sub>(n) passes, for pass p starting at p = 1, compute partial sums of n - 2<sup>p - 1</sup> in parallel
3. **GPU Work-Efficient Algorithm:**  O(n) operations - performs scan into two phases: parallel upsweep (reduction) with n - 1 adds (O(n)), and parallel downsweep with n - 1 adds (O(n)) and n - 1 swaps (O(n))
4. **GPU Naive Algorithm with Hardware Efficiency:**  shared memory - divide the array into evenly-sized blocks, each of which is scanned by a single thread block. Utilize shared memory within each thread block to perform the scan and write the total sum of each block to a separate array of block sums. Then, scan the array of block sums to create an array of block increments, which are added to all elements within their respective blocks.
5. **GPU Work-Efficient Algorithm with Hardware Efficiency:**  shared memory - same process as above
6. **GPU using [Thrust CUDA library](https://nvidia.github.io/cccl/thrust):**  wrapper function using thrust::exclusive_scan
