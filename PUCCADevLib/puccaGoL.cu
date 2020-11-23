/*
* Author: Juan Jose Olivera Loyola
* Name: PUCCA CA Implementation Template
* Description:
* This is a code template for programming parallelized cellular automatas.
* Usage:
* Just copy all of the code and change as needed.
* There are sections that are marked as @Customize where you are suggested to define your own data and logic rules.
*
* In this code template you will find an implementation for the Game of Life (GoL) cellular automata using
* as update function 'f' a variant working with convolution kernels.
*
*/
#include "puccaGoL.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// PUCCA Helper Libraries
#include "mat_utils.h"

/*
* CA Namesace
* @Customize: Change the name of the namespace to the name of your own cellular automata
*/
namespace GoLCA {

	/*
	* kCAStep kernel function declaration!
	* @Customize
	*	Keep it up to date with below definition!
	*/
	__global__ void kCAStep(int* prevState, int* nextState, int CA_N, int EXT_CA_N, int* GoLKernel);
	/* hCAReady function declaration */

	/*
	* kCellUpdate
	* It updates a single cell state as a function from the state of its neighbors
	* @Customization:
	*	User is free to define whatever logic and rule needed.
	* @BaseParameters:
	*	T* prevState (prevState data of @custom type T)
	*	T* nextState (nextState data of @custom type T)
	*	int CA_N (Original CA N)
	*	int EXT_CA_N (Extended CA N)
	*	int CA_N
	*	int EXT_CA_N
	* @UserDefinedParameters
	*	int* GoLKernel for performing a convolution and computing neighbors states
	*/
	__device__ void kCellUpdate(int col, int row, int* prevState, int* nextState, int CA_N, int EXT_CA_N, int* GoLKernel)
	{
		const int K_N = 3;
		int conv_index = (col + 1) + (row + 1) * EXT_CA_N;
		int aliveNeighbors = 0;

		for (int k_row = 0; k_row < K_N; k_row++) {
			for (int k_col = 0; k_col < K_N; k_col++) {
				aliveNeighbors +=
					GoLKernel[k_col + (k_row * K_N)] * prevState[(col + k_col) + (row + k_row) * EXT_CA_N];
			}
		}

		int cellValue = prevState[conv_index];
		cellValue = (1 - cellValue) * (aliveNeighbors == 3) + (cellValue) * (aliveNeighbors == 2 || aliveNeighbors == 3);
		nextState[conv_index] = cellValue;
	}


	/*
	* CA Data used for computing CA on GPU
	*/
	// Input Data
	int* caExtInitMat = NULL, * caFinalMat = NULL;
	int EXT_CA_N, CA_N;
	// Executing Configuration
	int BLOCKS_N, THREADS_N;
	// Internal Device and Host memory needed
	int* d_caExtMat = NULL, * d_caExtNewMat = NULL;
	int* d_tempPtr = NULL, * d_prevMat = NULL, * d_newMat = NULL;
	int* h_caExtNewMat = NULL;
	/* @Customization: Declare your custom extra variables needed in the computation that will need a GPU equivalent! */
	int* d_golKernel = NULL;
	// Benchmarking variables
	clock_t t1, t2;


	/*
	* hCAInit
	* It creates required buffer host memory for augmented CA and allocates the memory required for the GPU
	* Additional Functionality:
	*	It Setups the GoLKernel used to compute kCellUpdate
	*/
	void hCAInit(int* _initGolMat, int* _golKernel, int* _golFinalMat, int _n, int _nBlocks, int _nThreads)
	{
		/* Internal Main Functionality */
		// Setting input data
		EXT_CA_N = _n + 2;
		CA_N = _n;
		caExtInitMat = _initGolMat;
		caFinalMat = _golFinalMat;
		// Setting execution parameters
		THREADS_N = _nThreads;
		BLOCKS_N = _nBlocks;
		// Allocating internal memory
		cudaMalloc((void**)&d_caExtMat, EXT_CA_N * EXT_CA_N * sizeof(int));
		cudaMalloc((void**)&d_caExtNewMat, EXT_CA_N * EXT_CA_N * sizeof(int));
		h_caExtNewMat = (int*)malloc(EXT_CA_N * EXT_CA_N * sizeof(int));
		// Copying default initial data
		hCAReady();
		// Additional CUDA running configuration
		cudaFuncSetCacheConfig(kCAStep, cudaFuncCachePreferL1);

		/* @Customize: Additional Functionality */
		// Setup GoLKernel
		cudaMalloc((void**)&d_golKernel, 3 * 3 * sizeof(int));
		cudaMemcpy(d_golKernel, _golKernel, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);
	}

	/*
	* hCAReady
	* It transfers latest data changes and starts timer
	* Should be called before running a hCAStep loop
	*/
	void hCAReady()
	{
		cudaMemcpy(d_caExtMat, caExtInitMat, EXT_CA_N * EXT_CA_N * sizeof(int), cudaMemcpyHostToDevice);
		d_prevMat = d_caExtMat;
		d_newMat = d_caExtNewMat;
		t1 = clock();
	}

	/*
	* kCAStep
	* It executes a single parallel iteration step over the automata cells.
	* @BaseParameters:
	*	T* prevState (prevState data of @custom type T)
	*	T* nextState (nextState data of @custom type T)
	*	int CA_N (Original CA N)
	*	int EXT_CA_N (Extended CA N)
	* @UserDefinedParameters:
	*	int* GoLKernel: for computing kCellUpdate
	*/
	__global__ void kCAStep(int* prevState, int* nextState, int CA_N, int EXT_CA_N, int* GoLKernel)
	{
		int col_b = threadIdx.x + blockIdx.x * blockDim.x;
		int row_b = threadIdx.y + blockIdx.y * blockDim.y;

		int col = col_b, row = row_b;

		while (col < CA_N) {
			while (row < CA_N) {
				// @Customization: Update kCellUpdate parameters as defined in the function!
				kCellUpdate(col, row, prevState, nextState, CA_N, EXT_CA_N, GoLKernel);
				row += blockDim.y * gridDim.y;
			}
			row = row_b;
			col += blockDim.x * gridDim.x;
		}

		if (row < CA_N && col < CA_N) {
			// @Customization: Update kCellUpdate parameters as defined in the function!
			kCellUpdate(col, row, prevState, nextState, CA_N, EXT_CA_N, GoLKernel);
		}
	}

	/*
	* hCAStep
	* It invokes kCAStep properly configured and manages device memory addresses as needed
	* @BaseParameters:
	*	T* prevState
	*	T* nextState
	*	int gridN
	*	int convN
	* @UserDefinedParameters:
	*	CA Global data
	*/
	void hCAStep()
	{
		// @Customization Call kernel with additional variables
		kCAStep << <dim3(BLOCKS_N, BLOCKS_N), dim3(THREADS_N, THREADS_N) >> > (d_prevMat, d_newMat, CA_N, EXT_CA_N, d_golKernel);
		cudaDeviceSynchronize();

		// Check for errors
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			fprintf(stderr, "KERNEL_ERROR: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// Swap state memory to be ready to iterate next step
		d_tempPtr = d_prevMat;
		d_prevMat = d_newMat;
		d_newMat = d_tempPtr;

		// Copy back memory to host to read it
		cudaMemcpy(h_caExtNewMat, d_prevMat, EXT_CA_N * EXT_CA_N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	/*
	* hCARead
	* It copies back the data from the extended gpu memory into golFinalMat
	*/
	void hCARead()
	{
		cudaMemcpy(h_caExtNewMat, d_prevMat, EXT_CA_N * EXT_CA_N * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < CA_N; i++) {
			for (int j = 0; j < CA_N; j++) {
				// @Customization: Update finalMat with the correct format and data needed
				caFinalMat[j + i * CA_N] = h_caExtNewMat[(j + 1) + (i + 1) * EXT_CA_N];
			}
		}
	}

	/*
	* hCADone
	* Copies back CA data from GPU into final mat and frees GPU memory
	*/
	void hCADone()
	{
		t2 = clock();

		hCARead();

		printf("Internal GPU Time: %f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

		if (d_caExtMat != NULL)
			cudaFree(d_caExtMat);
		if (d_caExtNewMat != NULL)
			cudaFree(d_caExtNewMat);
		if (h_caExtNewMat != NULL)
			free(h_caExtNewMat);
		/* @Customization: Free your own specific memory! */
		if (d_golKernel != NULL)
			cudaFree(d_golKernel);
	}

};