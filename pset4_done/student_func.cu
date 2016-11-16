//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

// Modules for exclusive scan
// These functions are expected to be right for arbitrary size input
__global__ void excScan(unsigned int* d_in, unsigned int* d_out, int size)
{
   //Blelloch implementation of exclusive scan
   extern __shared__ int temp[];
   int thid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;
   int n = blockDim.x * 2;

   if (2*thid < size)
     temp[2*tid] = d_in[2 * thid];
   else
     temp[2*tid] = 0;

   if (2*thid + 1 < size)
     temp[2*tid + 1] = d_in[2*thid + 1];
   else
     temp[2*tid + 1] = 0;

   // Upsweeping
   int offset = 1;
   for (int d = n >> 1; d>0; d >>= 1)
   {
     if (tid < d)
     {
       int a = offset * (2*tid + 1) - 1;
       int b = offset * (2*tid + 2) - 1;
       temp[b] += temp[a];
     }
     __syncthreads();
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
 
   if (2*thid < size)  
     d_out[2*thid] = temp[2*tid];
   if (2*thid + 1 < size)
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

// Histogram Algorithm
__global__ void histogramKernel(unsigned int* zero, unsigned int* d_histo, size_t numElems)
{
   int thid = threadIdx.x + blockIdx.x * blockDim.x;
   int bin = 0;
   if (thid < numElems)
   {
      bin = 1 - zero[thid];
      atomicAdd(&d_histo[bin], 1); 
   }
   __syncthreads();
}

void parallelHistogram(unsigned int* zero, unsigned int* d_histo, size_t numElems)
{
   size_t K = 512;
   dim3 gridSize((numElems + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);

   histogramKernel<<<gridSize, blockSize>>>(zero, d_histo, numElems);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}

// Generating one and zero vector
__global__ void generateKernel(const unsigned int* d_inputVals, unsigned int* zero, 
			       unsigned int* one, size_t numElems, unsigned int shiftBit, int i)
{
   int thid = threadIdx.x + blockIdx.x * blockDim.x;
   if (thid < numElems)
   {
      unsigned int isOne = (d_inputVals[thid] & shiftBit) >> i;
      one[thid] = isOne;
      zero[thid] = 1 - isOne;
   }
}

void generateZeroOne(const unsigned int* d_inputVals, 
                     unsigned int* zero, 
                     unsigned int* one, 
                     size_t numElems, 
                     int i)
{
   unsigned int shiftBit = 0x1 << i;
   
   size_t K = 512;
   dim3 gridSize((numElems + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);

   generateKernel<<<gridSize, blockSize>>>(d_inputVals, zero, one, numElems, shiftBit, i);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}

// Move Elements
__global__ void moveKernel(unsigned int* d_inputVals, unsigned int* d_inputPos,
			unsigned int* d_outputVals, unsigned int* d_outputPos,
			size_t numElems, unsigned int* zero,
			unsigned int* scanZero, unsigned int* scanOne,
			int zeroCount)
{
   int thid = threadIdx.x + blockIdx.x * blockDim.x;
   if (thid < numElems)
   {
      int isZero = zero[thid];
      int isOne = 1 - isZero;
      int outPosition = isOne * (scanOne[thid] + zeroCount) + isZero * scanZero[thid];
      d_outputVals[outPosition] = d_inputVals[thid];
      d_outputPos[outPosition] = d_inputPos[thid]; 
   }
   __syncthreads();
}

void moveElements(unsigned int* scanZero, unsigned int* scanOne, 
                  unsigned int* d_histo, unsigned int* zero,
                  unsigned int* d_inputVals, unsigned int* d_inputPos, 
                  unsigned int* d_outputVals, unsigned int* d_outputPos, 
                  size_t numElems)
{
   size_t K = 512;
   dim3 gridSize((numElems + K - 1) / K, 1, 1);
   dim3 blockSize(K, 1, 1);
 
   unsigned int zeroCount;
   checkCudaErrors(cudaMemcpy(&zeroCount, d_histo, sizeof(unsigned), cudaMemcpyDeviceToHost));
   //printf("zeros = %u size = %lu\n", zeroCount, numElems);
   moveKernel<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, 
                                       d_outputVals, d_outputPos, 
                                       numElems, zero,
                                       scanZero, scanOne, 
                                       zeroCount);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}



// API
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  unsigned int* scanZero;
  unsigned int* scanOne;
  unsigned int* zero;
  unsigned int* one;
  unsigned int* updatePosition;
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * 2));
  checkCudaErrors(cudaMalloc(&scanZero, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&scanOne, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&zero, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&one, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&updatePosition, sizeof(unsigned int) * numElems));

  for (int i=0; i<32; i++)
  {
     generateZeroOne(d_inputVals, zero, one, numElems, i);
     parallelScan(zero, scanZero, numElems);
     parallelScan(one, scanOne, numElems);
     parallelHistogram(zero, d_histo, numElems);
     moveElements(scanZero, scanOne, d_histo, zero, d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
     
     if (i != 31)
     { 
       checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
       checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
     }
     checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * 2));
  }
}
