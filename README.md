CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Michael Rabbitz
  * [LinkedIn](https://www.linkedin.com/in/mike-rabbitz)
* Tested on: Windows 10, i7-9750H @ 2.60GHz 32GB, RTX 2060 6GB (Personal)

## Part 1: Introduction

This project focuses on implementing various Stream Compaction algorithms, including those utilizing the Scan algorithm, to emphasize the importance of designing GPU hardware-optimized algorithms that leverage parallel computation for superior performance compared to CPU implementations.

### Stream Compaction
Stream compaction involves filtering an input data set to produce a new collection that contains only the elements meeting a specified condition, while preserving their original order. This process reduces the size of the data set by removing unwanted elements, which is crucial for optimizing performance and memory usage in applications like path tracing, collision detection, etc.

<p align="left">
  <img src="img/stream_compaction_visual.PNG" />
</p>

### Scan
The Scan algorithm, also known as the all-prefix-sums operation, computes prefix sums for an array of data. In an exclusive scan, each element in the result is the sum of all preceding elements in the input array. Conversely, an inclusive scan includes the current element in the sum. Scan algorithms are fundamental in parallel computing for tasks like sorting, stream compaction, and building data structures, as they transform inherently sequential computations into parallel ones.

<p align="left">
  <img src="img/scan_visual.PNG" />
</p>

## Part 2: Implementation Details

**n represents the number of elements in the array**

**All implementations support arrays of arbitrary n - small, large, powers of two, not powers of two**

### Scan
#### Implementations
1. **CPU:**  O(n) addition operations - sequential loop over array elements, accumulating a sum at each iteration
2. **GPU Naive Algorithm:**  O(n * log<sub>2</sub>(n)) addition operations - over log<sub>2</sub>(n) passes, for pass p starting at p = 1, compute the partial sums of n - 2<sup>p - 1</sup> elements in parallel
3. **GPU Work-Efficient Algorithm:**  O(n) operations - performs scan into two phases: parallel upsweep (reduction) with n - 1 adds (O(n)), and parallel downsweep with n - 1 adds (O(n)) and n - 1 swaps (O(n))
4. **GPU Naive Algorithm with Hardware Efficiency:**  shared memory - divide the array into evenly-sized blocks, each of which is scanned by a single thread block. Utilize shared memory within each thread block to perform the scan and write the total sum of each block to a separate array of block sums. Then, scan the array of block sums to create an array of block increments, which are added to all elements within their respective blocks.
5. **GPU Work-Efficient Algorithm with Hardware Efficiency:**  shared memory - same process as above
6. **GPU using [Thrust CUDA library](https://nvidia.github.io/cccl/thrust):**  wrapper function using thrust::exclusive_scan

#### Inclusive Scan - GPU Naive Algorithm:
<p align="left">
  <img src="img/gpu_inclusive_naive.PNG" />
</p>

#### Exclusive Scan - GPU Work-Efficient Algorithm:
**Upsweep**
<p align="left">
  <img src="img/gpu_exclusive_efficient_upsweep.PNG" />
</p>

**Downsweep**
<p align="left">
  <img src="img/gpu_excluisve_efficient_downsweep.PNG" />
</p>

### Stream Compaction
**Compaction removes invalid elements (0s) from an array of randomized ints**

#### Stream Compaction with Scan is described in 3 steps:
1. **Create Binary Map:** use the input array to create a binary map array indicating the validity of each input element
2. **Scan:** perform Scan on the binary array to generate a map of index values which correspond to the compacted array index of sequential valid input elements
3. **Scatter:** for the indices of the binary array that indicate valid input elements, use those indices to index the Scan output array to determine the compated array index of the valid element, then place valid element data into the compacted array

<p align="left">
  <img src="img/compaction_with_scan.PNG" />
</p>

#### Implementations
1. **CPU without Scan:** - sequential loop over input elements while placing valid input data into the compacted array
2. **CPU with Scan:** - perform Step 1 with sequential loop over n elements, perform Step 2 using CPU Scan, then perform Step 3 with sequential loop over n elements
3. **GPU with Work-Efficient Scan:** - perform Step 1 over n elements in one parallel pass, perform Step 2 using Work-Efficient Scan, then perform Step 3 over n elements in one parallel pass
4. **GPU with Work-Efficient and Hardware-Efficient Scan:** - same as above line except using Work-Efficient and Hardware-Efficient Scan
5. **GPU using [Thrust CUDA library](https://nvidia.github.io/cccl/thrust):**  wrapper function using thrust::remove_if

## Part 3: Performance Analysis
### Optimal Block Size
**For each GPU Scan implementation, find the block size that gives the minimal runtime on the GPU**

**Each data point is the average of 100 runtime samples**
<p align="left">
  <img src="img/runtime_vs_blocksize.png" />
</p>

| Block Size | Naive (ms) | Naive & Hardware-Eff (ms) | Work-Eff | Work-Eff & Hardware-Eff |
| ---------- | ---------- | ------------------------- | -------- | ----------------------- |
|32          |191.8       |24                         |47.6      |13.1                     |
|64          |116.9       |14.4                       |47.9      |9.7                      |
|128         |116.8       |13.8                       |47.4      |9.3                      |
|256         |116.8       |14                         |47.8      |9.6                      |
|512         |116.8       |14.4                       |48.3      |10.5                     |
|1024        |118.1       |16.6                       |50.4      |12.4                     |

<h1 style="font-size: 24px;">Performance Comparison</h1>

<table style="font-size: 14px;">
  <tr><th>Block Size</th><th>Naive (ms)</th><th>Naive & Hardware-Eff (ms)</th><th>Work-Eff</th><th>Work-Eff & Hardware-Eff</th></tr>
  <tr><td>32</td><td>191.8</td><td>24</td><td>47.6</td><td>13.1</td></tr>
  <tr><td>64</td><td>116.9</td><td>14.4</td><td>47.9</td><td>9.7</td></tr>
  <tr><td>128</td><td>116.8</td><td>13.8</td><td>47.4</td><td>9.3</td></tr>
  <tr><td>256</td><td>116.8</td><td>14</td><td>47.8</td><td>9.6</td></tr>
  <tr><td>512</td><td>116.8</td><td>14.4</td><td>48.3</td><td>10.5</td></tr>
  <tr><td>1024</td><td>118.1</td><td>16.6</td><td>50.4</td><td>12.4</td></tr>
</table>

Based on the results, it is clear that block size of 128 is optimal.

### Runtime
**This includes the CPU Scan implementation, the GPU Scan implementations, and the Thrust Scan**

**Each data point is the average of 100 runtime samples**
<p align="left">
  <img src="img/runtime_vs_arraysize.png" />
</p>

|  Array Size  | CPU (ms) | Naive (ms)  | Naive & Hardware-Eff (ms)  | Work-Eff  | Work-Eff & Hardware-Eff  | Thrust (ms)  |
| ------------ | -------- |------------ | -------------------------- | --------- | ------------------------ | ------------ |
|2<sup>10</sup>|0.0005    |0.0332       |0.0626                      |0.0786     |0.0742                    |0.0670        |
|2<sup>12</sup>|0.0018    |0.0390       |0.0570                      |0.1122     |0.0766                    |0.1143        |
|2<sup>14</sup>|0.0068    |0.0494       |0.0589                      |0.1236     |0.0598                    |0.0514        |
|2<sup>16</sup>|0.0241    |0.0756       |0.1145                      |0.1271     |0.0776                    |0.0793        |
|2<sup>18</sup>|0.1092    |0.1705       |0.2576                      |0.1742     |0.1362                    |0.1640        |
|2<sup>20</sup>|0.5457    |0.8490       |0.3955                      |0.4744     |0.2322                    |0.2773        |
|2<sup>22</sup>|2.8498    |2.9690       |0.7290                      |1.4791     |0.4043                    |0.3592        |
|2<sup>24</sup>|13.7399   |12.8068      |1.9275                      |5.9673     |1.2438                    |0.6996        |

**Look at Thrust Implementation**

### Performance Bottlenecks

### Output



