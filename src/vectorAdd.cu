/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <sys/time.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(int argc, char** argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    unsigned long long numElements = 2<<20	; // <------- size!
    int threadsPerBlock = 256, option;

    while ((option = getopt(argc, argv, "n:t:")) != -1) {
    	switch (option) {
    		case 'n':
    			numElements = (unsigned long long) atoi(optarg);
    			break;
    		case 't':
    			threadsPerBlock = (int) atoi(optarg);
    			break;
    		default:
    			printf("Error!\n");
    			exit(0);
    	}
    }

    size_t size = numElements * sizeof(float);

	cudaDeviceProp prop;
	int numDevices = 0;

	err = cudaGetDeviceCount(&numDevices);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to query the number of devices!\n");
		exit(EXIT_FAILURE);
	}

	int totalMem = 0;

	for (int i = 0; i < numDevices; i++) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to query the device properties!\n");
			exit(EXIT_FAILURE);
		}

		totalMem += prop.totalGlobalMem;
	}

	if (size > totalMem){
		printf("Memory exceeded!\n");
		exit(EXIT_FAILURE);
	}

    printf("[Vector addition of %d elements]\n", numElements);

    float* A, *B, *C;
    err = cudaMallocManaged(&A, size);
    if (err != cudaSuccess) {
    	fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged(&B, size);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

    err = cudaMallocManaged(&C, size);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }

	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
    cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

