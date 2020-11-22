#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mat_utils.h"

namespace GoLCA{

__global__ void kCAStep(int* prevConvMat, int* newConvMat, int* kernel, int n, int nConv);


class MUtil
{
public:
	int m;
	__host__ __device__ MUtil() {}
	__host__ __device__ int conv2dto1d(int x, int y) {
		return x + y*m;
	}
};

class ExternalDeviceCall
{
	__device__ void golConvolution(int col, int row, int* d_kernel, int k_size, int* prevMat, int convN, int* newMat, int n)
	{
		//MUtil util;
		int conv_index = (col + 1) + (row + 1) * convN;
		//util->conv2dto1d(col + 1, row + 1);

		int aliveNeighbors = 0;

		for (int k_row = 0; k_row < k_size; k_row++) {
			for (int k_col = 0; k_col < k_size; k_col++) {
				aliveNeighbors +=
					d_kernel[k_col + (k_row * k_size)] * prevMat[(col + k_col) + (row + k_row) * convN];
			}
		}

		int cellValue = prevMat[conv_index];
		cellValue = (1 - cellValue) * (aliveNeighbors == 3) + (cellValue) * (aliveNeighbors == 2 || aliveNeighbors == 3);
		newMat[conv_index] = cellValue;
	}
};


__device__ void kCellUpdate(int col, int row, int* d_kernel, int k_size, int* prevMat, int convN, int* newMat, int n)
{
	int conv_index = (col + 1) + (row + 1) * convN;
	int aliveNeighbors = 0;

	for (int k_row = 0; k_row < k_size; k_row++) {
		for (int k_col = 0; k_col < k_size; k_col++) {
			aliveNeighbors +=
				d_kernel[k_col + (k_row * k_size)] * prevMat[(col + k_col) + (row + k_row) * convN];
		}
	}

	int cellValue = prevMat[conv_index];
	cellValue = (1 - cellValue) * (aliveNeighbors == 3) + (cellValue) * (aliveNeighbors == 2 || aliveNeighbors == 3);
	newMat[conv_index] = cellValue;
}

clock_t t1, t2;
int CONV_N, CA_N, BLOCKS_N, THREADS_N;
int* golFinalMat, * golInitMat;
int* h_newConvMat;
int* d_convMat, * d_newConvMat, * d_golKernel;
int* d_tempPtr, * d_prevMat, * d_newMat;

/*
* hCAInit
* It creates required buffer host memory for augmented CA and allocates the memory required for the GPU
* Additional Functionality:
*	It Setups the GoLKernel used to compute kCellUpdate
*/
void hCAInit(int* _initGolMat, int* _golKernel, int* _golFinalMat, int _n, int _nBlocks, int _nThreads)
{
	CONV_N = _n + 2;
	CA_N = _n;
	THREADS_N = _nThreads;
	BLOCKS_N = _nBlocks;

	golInitMat = _initGolMat;
	golFinalMat = _golFinalMat;

	cudaMalloc((void**)&d_convMat, CONV_N * CONV_N * sizeof(int));
	cudaMalloc((void**)&d_newConvMat, CONV_N * CONV_N * sizeof(int));
	
	h_newConvMat = (int*)malloc(CONV_N * CONV_N * sizeof(int));

	cudaMemcpy(d_convMat, golInitMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyHostToDevice);

	d_prevMat = d_convMat;
	d_newMat = d_newConvMat;

	cudaFuncSetCacheConfig(kCAStep, cudaFuncCachePreferL1);

	/* Additional Functionality */
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
	cudaMemcpy(d_convMat, golInitMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyHostToDevice);
	d_prevMat = d_convMat;
	d_newMat = d_newConvMat;
	t1 = clock();

	printf("CA Init Mat\n");
	printMat(golInitMat, CONV_N);
}

/*
* kCAStep
* It executes a single parallel iteration step over the automata cells.
* @BaseParameters:
*	T* prevState
*	T* nextState
*	int gridN
*	int convN
* @UserDefinedParameters:
*	CA Global data
*/
__global__ void kCAStep(int* prevConvMat, int* newConvMat, int* kernel, int n, int nConv)
{
	int col_b = threadIdx.x + blockIdx.x * blockDim.x;
	int row_b = threadIdx.y + blockIdx.y * blockDim.y;

	int col = col_b, row = row_b;

	while (col < n) {
		while (row < n) {
			kCellUpdate(col, row, kernel, 3, prevConvMat, nConv, newConvMat, n);
			row += blockDim.y * gridDim.y;
		}
		row = row_b;
		col += blockDim.x * gridDim.x;
	}

	if (row < n && col < n) {
		kCellUpdate(col, row, kernel, 3, prevConvMat, nConv, newConvMat, n);
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
void hCAStep(int t)
{

	if (t % 1 == 0) printf("Iteracion #%d\n", t);
	kCAStep << <dim3(BLOCKS_N, BLOCKS_N), dim3(THREADS_N, THREADS_N) >> > (d_prevMat, d_newMat, d_golKernel, CA_N, CONV_N);
	//(*kFunc) << <dim3(nBlocks, nBlocks), dim3(nThreads, nThreads) >> > (d_prevMat, d_newMat, d_golKernel, n, CONV_N);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: validClassKernel: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	d_tempPtr = d_prevMat;
	d_prevMat = d_newMat;
	d_newMat = d_tempPtr;

	cudaMemcpy(h_newConvMat, d_prevMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyDeviceToHost);
}

/*
* hCADone
* Copies back CA data from GPU into final mat and frees GPU memory
*/
void hCADone()
{
	cudaMemcpy(h_newConvMat, d_prevMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyDeviceToHost);
	t2 = clock();


	for (int i = 0; i < CA_N; i++) {
		for (int j = 0; j < CA_N; j++) {
			golFinalMat[j + i * CA_N] = h_newConvMat[(j + 1) + (i + 1) * CONV_N];
		}
	}

	printf("Internal GPU Time: %f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	cudaFree(d_convMat);
	cudaFree(d_newConvMat);
	cudaFree(d_golKernel);
	free(h_newConvMat);
}

};