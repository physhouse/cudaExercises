/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define MIN_POSSIBLE -1.0E-30
#define MAX_POSSIBLE 9.0E30

__global__ void shmemMinMaxReducePerBlock(const float* d_in,
                                          const size_t arraySize,
                                          float* d_out,
					  bool  isMax)
{
   extern __shared__ float shared[];
   int tid = threadIdx.x;
   int myId = threadIdx.x + blockDim.x * blockIdx.x;

   // Loading the global memory to shared memory
   if (2 * myId < arraySize)
   {
     shared[2*tid] = d_in[2*myId];
   }
   else
   {
     if (isMax)
        shared[2*tid] = MIN_POSSIBLE;
     else
        shared[2*tid] = MAX_POSSIBLE;
   }
  
   if (2 * myId + 1 < arraySize)
   {
     shared[2*tid + 1] = d_in[2*myId + 1];
   }
   else
   {
     if (isMax)
        shared[2*tid + 1] = MIN_POSSIBLE;
     else
        shared[2*tid + 1] = MAX_POSSIBLE;
   }
   __syncthreads();

   // Parallel Reduce
   for (size_t step = blockDim.x; step > 0; step /= 2)
   {
     if (tid < step)
     {
       if (isMax)
         shared[tid] = max(shared[tid], shared[tid + step]);
       else
         shared[tid] = min(shared[tid], shared[tid + step]);
     }
     __syncthreads();
   }
 
   if (tid == 0)
   {
      d_out[blockIdx.x] = shared[0];        
   }
}

__global__ void histogramKernel(const float* const d_in, unsigned int* d_histo, int arraySize, float minVal, float range, int numBins)
{
   int myId = threadIdx.x + blockDim.x * blockIdx.x;
   if (myId < arraySize)
   {
      int binId = (d_in[myId] - minVal) / range * numBins;
      atomicAdd(&d_histo[binId], 1);
   }
}

__global__ void excScan(unsigned int* d_in, unsigned int* d_out, int n)
{
   //Blelloch implementation of exclusive scan
   extern __shared__ int temp[];
   int thid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;

   temp[2*tid] = d_in[2 * thid];
   temp[2*tid + 1] = d_in[2*thid + 1];

   int offset = 1;
   for (int d = n >> 1; d>0; d >>= 1)
   {
     __syncthreads();
     if (tid < d)
     {
       int a = offset * (2*tid + 1) - 1;
       int b = offset * (2*tid + 2) - 1;
       temp[b] += temp[a];
     }
     offset *= 2;
   }
   // Downsweeping
   if (tid == 0) temp[n-1] = 0;

   offset = n;
   for (int d = 1; d<n; d *= 2)
   {
      __syncthreads();
      if (tid < d)
      {
         int a = n - tid * offset - 1;
         int b = n - tid * offset - 1 - offset / 2;
         int tmp = temp[a];
         temp[a] += temp[b];
         temp[b] = tmp;
      }
      offset >>= 1;
   }
   __syncthreads();
   
   d_out[2*thid] = temp[2*tid];
   d_out[2*thid + 1] = temp[2*tid + 1];
}

__global__ void preSum(unsigned int* d_in, unsigned int* d_out, unsigned int* sumBlocks, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //calculate the sum of all numbers in the bid'th block and save it in sumBlocks
    if (tid == 0) sumBlocks[bid] = d_in[n*(bid+1) -1] + d_out[n*(bid+1) -1];
}

__global__ void postSum(unsigned int* d_out, unsigned int* scanBlocks, int length)
{
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int bid = blockIdx.x;

    d_out[2*thid] += scanBlocks[bid];
    d_out[2*thid+1] += scanBlocks[bid];
}

static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

//Recursive parallel scan, this make sure that this algorithm works for arrays with any length
// Using a sumblock[] and scanblock[] arrays to faccilitate this
void parallelScan(unsigned int* d_in, unsigned int* d_out, size_t size)
{
   int K = 512;
   while (size < 2 * K)
     K /= 2;

   dim3 gridSize(((size+1)/2 + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);
   int bytesShared = sizeof(int) * 2 * K;

   unsigned int* sumBlocks;
   unsigned int* scanBlocks;
   int sizeNext = nextPow2(gridSize.x);
   checkCudaErrors(cudaMalloc(&sumBlocks, sizeof(int) * sizeNext));
   checkCudaErrors(cudaMalloc(&scanBlocks, sizeof(unsigned int) * sizeNext));
   checkCudaErrors(cudaMemset(sumBlocks, 0, gridSize.x));
   checkCudaErrors(cudaMemset(scanBlocks, 0, gridSize.x));

   excScan<<<gridSize, blockSize, bytesShared>>>(d_in, d_out, size);

   if (gridSize.x > 1)
   {
       preSum<<<gridSize, blockSize>>>(d_in, d_out, sumBlocks, 2*K);
       
       parallelScan(sumBlocks, scanBlocks, gridSize.x);

       postSum<<<gridSize, blockSize>>>(d_out, scanBlocks, 2*K);
   }
   
   checkCudaErrors(cudaFree(sumBlocks));
   checkCudaErrors(cudaFree(scanBlocks));  
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}

__global__ void inclusiveKernel(unsigned int* d_in, unsigned int* d_out, size_t size)
{
   int thid = threadIdx.x + blockDim.x * blockIdx.x;
   if (thid < size)
   {
      d_out[thid] = d_out[thid] + d_in[thid];
   }
   __syncthreads();
}

void parallelInclusiveScan(unsigned int* d_in, unsigned int* d_out, size_t size)
{
   parallelScan(d_in, d_out, size);
 
   int K = 512;

   dim3 gridSize((size + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);
   inclusiveKernel<<<gridSize, blockSize>>>(d_in, d_out, size);
}

__global__ void checkHistogram(unsigned int* d_histo)
{
   printf("bin%d: %u\n", threadIdx.x, d_histo[threadIdx.x]);
}

void parallelHistogram(const float* const d_in,
                       unsigned int* d_histo,
                       int arraySize,
                       float minVal,
                       float maxVal,
                       int numBins)
{
   int K = 512;
   dim3 gridSize((arraySize + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);
   float range = maxVal - minVal;

   histogramKernel<<<gridSize, blockSize>>>(d_in, d_histo, arraySize, minVal, range, numBins);
   //checkHistogram<<<1, numBins>>>(d_histo);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}

float parallelMinMaxReduce(const float* const d_in, 
			  const size_t arraySize, 
			  bool  isMax)
{
  int K = 512;
  int sizeThreads = (arraySize + 1) / 2;
  dim3 gridSize((sizeThreads + K - 1) / K, 1, 1);
  dim3 blockSize(K, 1, 1);

  float* d_curr_out;
  float* d_curr_in;
  size_t current_size = arraySize;

  checkCudaErrors(cudaMalloc(&d_curr_in, sizeof(float) * arraySize));
  checkCudaErrors(cudaMemcpy(d_curr_in, d_in, sizeof(float) * arraySize, cudaMemcpyDeviceToDevice));

  while (current_size > 1)
  {
    checkCudaErrors(cudaMalloc(&d_curr_out, sizeof(float) * gridSize.x));
    shmemMinMaxReducePerBlock<<<gridSize, blockSize, sizeof(float) * 2 * K>>>(d_curr_in, current_size, d_curr_out, isMax);

    checkCudaErrors(cudaFree(d_curr_in));
    d_curr_in = d_curr_out;

    current_size = gridSize.x;
    gridSize.x = ((gridSize.x + 1)/2 + K - 1) / K;
  }

  float h_output;
  checkCudaErrors(cudaMemcpy(&h_output, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_curr_out);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  return h_output;
  
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
				  unsigned int* const d_cdf,
				  float &min_logLum,
				  float &max_logLum,
				  const size_t numRows,
				  const size_t numCols,
				  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  // Step 1, parallel Minmax reduce
  size_t arraySize = numRows * numCols;
  max_logLum =  parallelMinMaxReduce(d_logLuminance, arraySize, 1);
  min_logLum =  parallelMinMaxReduce(d_logLuminance, arraySize, 0);
  //printf("max = %lf, min = %lf\n", max_logLum, min_logLum);
 
  // Step 2, parallel Histogram using atomic operations
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  parallelHistogram(d_logLuminance, d_histo, arraySize, min_logLum, max_logLum, numBins);

  // Step 3, parallel Scan to accumulate the histogram
  parallelScan(d_histo, d_cdf, numBins);

  checkCudaErrors(cudaFree(d_histo));

}

