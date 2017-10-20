//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <climits>
#include <math.h>
#include <stdio.h>
#include <iostream>
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


__global__
void RadixSort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	extern __shared__ unsigned int shdata[];
	unsigned int mask = 1;
	int upperLimit = int(log2(float(UINT_MAX)));
	int tid = threadIdx.x;
	int myID = blockDim.x * blockIdx.x + threadIdx.x;
	if (myID >= numElems){
		return;
	}
	for (int i = 0; i < upperLimit; ++i){
		if (i>0){
			mask<<=1;
		}
		// get the corresponding bit
		// from input to output

		

		if (i%2 == 0){
			unsigned int digit = d_inputVals[myID] & mask;
			// calculate prefix sum

			// 1. copy the data
			d_outputPos[myID] = digit;
			__syncthreads();
			// 2. do prefix sum -> code for blelloch scan


		  // load data
		  shdata[tid] = digit;
		  __syncthreads();

		  // first half: reduction
		  
		  for (size_t s = (blockDim.x >>1); s > 0; s>>=1){
		    if (tid < s){
		      int stride = blockDim.x / s;
		      int index = (tid+1)*stride -1;
		      shdata[index] = shdata[index] + shdata[(index - (stride>>1))];
		    }
		    __syncthreads();
		    
		  }
		  

		  // make the last element to become zero
		  if (tid == blockDim.x - 1){
		    shdata[tid] = 0;
		  }
		  __syncthreads();

		  // second half: post reduction
		  for (size_t s = 1; s < blockDim.x; s<<=1){
		    if (tid < s){
		      int stride = (blockDim.x)/s;
		      int index = (tid+1)*stride - 1;
		      unsigned int smallIdxVal = shdata[index];
		      unsigned int largeIdxVal = shdata[index] + shdata[index - (stride>>1)]; 
		      shdata[index-(stride>>1)] = smallIdxVal;
		      shdata[index] = largeIdxVal;
		    }
		    __syncthreads();

		  }

		  // copy to the result array
		  d_outputPos[myID] = shdata[tid];
		  __syncthreads();

		  // deal with different block
		  unsigned int offset = 0;
		  if (blockIdx.x > 0){
		    for (int i = 0; i < blockIdx.x; ++i){
		      // note here we do not store back data
		      // offset += 267;// max(offset, d_outputPos[(i+1)*blockDim.x - 1]);
		      offset =  d_outputPos[(i+1)*blockDim.x - 1];
		      // if (i>tid){
		      // 	break;
		      // }
		    }
		    __syncthreads();

	    	d_outputPos[myID] = offset;
			
		  	__syncthreads();
			return;

			// get prefix sum at output_Vals

		  	// histogram of 0 and 1
			if (digit % 2 == 0){
				atomicAdd(&d_outputVals[0], 1);
			}
			__syncthreads();
			if (digit % 2 == 1){
				d_outputPos[myID] += d_outputVals[0];
			}
			__syncthreads();
			// move data	
			// d_outputVals[d_outputPos[myID]] = d_inputVals[myID];
			// d_outputPos[d_outputPos[myID]]  = d_inputPos[myID];
			__syncthreads();

		}
		// from output to input
		else{
			unsigned int digit = d_outputVals[myID] & mask;
			// calculate prefix sum

			// 1. copy the data
			d_inputPos[myID] = digit;
			__syncthreads();
			// 2. do prefix sum -> code for blelloch scan


		  // load data
		 //  shdata[tid] = d_inputPos[myID];
		 //  __syncthreads();

		 //  // first half: reduction
		  
		 //  for (size_t s = (blockDim.x >>1); s > 0; s>>=1){
		 //    if (tid < s){
		 //      int stride = blockDim.x / s;
		 //      int index = (tid+1)*stride -1;
		 //      shdata[index] = shdata[index] + shdata[(index - (stride>>1))];
		 //    }
		 //    __syncthreads();
		    
		 //  }
		  

		 //  // make the last element to become zero
		 //  if (tid == blockDim.x - 1){
		 //    shdata[tid] = 0;
		 //  }
		 //  __syncthreads();

		 //  // second half: post reduction
		 //  for (size_t s = 1; s < blockDim.x; s<<=1){
		 //    if (tid < s){
		 //      int stride = (blockDim.x)/s;
		 //      int index = (tid+1)*stride - 1;
		 //      unsigned int smallIdxVal = shdata[index];
		 //      unsigned int largeIdxVal = shdata[index] + shdata[index - (stride>>1)]; 
		 //      shdata[index-(stride>>1)] = smallIdxVal;
		 //      shdata[index] = largeIdxVal;
		 //    }
		 //    __syncthreads();

		 //  }

		 //  // copy to the result array
		 //  d_inputPos[myID] = shdata[tid];
		 //  __syncthreads();

		 //  // deal with different block
		 //  unsigned int offset = 0;
		 //  if (blockIdx.x > 0){
		 //    for (int i = 0; i < blockIdx.x; ++i){
		 //      // note here we do not store back data
		 //      offset += d_inputPos[(i+1)*blockDim.x - 1];
		 //    }
		 //    __syncthreads();
		 //    d_inputPos[myID] = d_inputPos[myID] + offset;
		 //  }

			// // get prefix sum at output_Vals

		 //  	// histogram of 0 and 1
			// if (digit % 2 == 0){
			// 	atomicAdd(&d_inputVals[0], 1);
			// }
			// __syncthreads();
			// if (digit % 2 == 1){
			// 	d_inputPos[myID] += d_inputVals[0];
			// }
			// __syncthreads();

			// // move data
			// d_inputVals[d_inputPos[myID]] = d_outputVals[myID];
			// d_inputPos[d_inputPos[myID]]  = d_outputPos[myID];
			// __syncthreads();
		}

	}

}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

	int blockSize = 512;
	int gridSzie = (numElems - 1)/blockSize + 1;
	RadixSort<<<gridSzie, blockSize, sizeof(unsigned int)*blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
	unsigned int * h_ouputPos = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	checkCudaErrors(cudaMemcpy(h_ouputPos, d_outputPos, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i<numElems; ++i){
		std::cout<< h_ouputPos[i] <<std::endl;
	}

}
