#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mat_utils.h"

__device__ void golConvolution(int col, int row, int* d_kernel, int k_size, int* prevMat, int convN, int* newMat, int n)
{
	int conv_index = (col + 1) + (row + 1) * convN;

	int aliveNeighbors = 0;

	aliveNeighbors += d_kernel[0 + (0 * k_size)] * prevMat[(col + 0) + (row + 0) * convN];
	aliveNeighbors += d_kernel[1 + (0 * k_size)] * prevMat[(col + 1) + (row + 0) * convN];
	aliveNeighbors += d_kernel[2 + (0 * k_size)] * prevMat[(col + 2) + (row + 0) * convN];
	aliveNeighbors += d_kernel[0 + (1 * k_size)] * prevMat[(col + 0) + (row + 1) * convN];
	aliveNeighbors += d_kernel[1 + (1 * k_size)] * prevMat[(col + 1) + (row + 1) * convN];
	aliveNeighbors += d_kernel[2 + (1 * k_size)] * prevMat[(col + 2) + (row + 1) * convN];
	aliveNeighbors += d_kernel[0 + (2 * k_size)] * prevMat[(col + 0) + (row + 2) * convN];
	aliveNeighbors += d_kernel[1 + (2 * k_size)] * prevMat[(col + 1) + (row + 2) * convN];
	aliveNeighbors += d_kernel[2 + (2 * k_size)] * prevMat[(col + 2) + (row + 2) * convN];


	/*for (int k_row = 0; k_row < k_size; k_row++) {
		for (int k_col = 0; k_col < k_size; k_col++) {
			aliveNeighbors +=
				d_kernel[k_col + (k_row * k_size)] * prevMat[(col + k_col) + (row + k_row) * convN];
		}
	}*/

	int cellValue = prevMat[conv_index];
	/*if (cellValue == 0) {
		if (aliveNeighbors == 3) cellValue = 1;
		else cellValue = 0;
	}
	else {
		if (aliveNeighbors == 2 || aliveNeighbors == 3) cellValue = 1;
		else cellValue = 0;
	}*/
	cellValue = (1 - cellValue) * (aliveNeighbors == 3) + (cellValue) * (aliveNeighbors == 2 || aliveNeighbors == 3);
	newMat[conv_index] = cellValue;
}

__global__ void kCalcGoLIteration(int* prevConvMat, int* newConvMat, int* kernel, int n, int nConv)
{
	int col_b = threadIdx.x + blockIdx.x * blockDim.x;
	int row_b = threadIdx.y + blockIdx.y * blockDim.y;

	int col = col_b, row = row_b;

	while (col < n) {
		while (row < n) {
			golConvolution(col, row, kernel, 3, prevConvMat, nConv, newConvMat, n);
			row += blockDim.y * gridDim.y;
		}
		row = row_b;
		col += blockDim.x * gridDim.x;
	}

	if (row < n && col < n) {
		golConvolution(col, row, kernel, 3, prevConvMat, nConv, newConvMat, n);
	}
}

void gpuPlayGoL(int* initGolMat, int* golKernel, int* golFinalMat, int n, int nSteps,  int nBlocks, int nThreads)
{
	clock_t t1, t2;
	int CONV_N = n + 2;
	int* h_newConvMat;
	int* d_convMat, * d_newConvMat, * d_golKernel;
	int* d_tempPtr, * d_prevMat, * d_newMat;

	cudaMalloc((void**)&d_convMat, CONV_N * CONV_N * sizeof(int));
	cudaMalloc((void**)&d_newConvMat, CONV_N * CONV_N * sizeof(int));
	cudaMalloc((void**)&d_golKernel, 3 * 3 * sizeof(int));
	h_newConvMat = (int*)malloc(CONV_N * CONV_N * sizeof(int));

	cudaMemcpy(d_convMat, initGolMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_golKernel, golKernel, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);

	d_prevMat = d_convMat;
	d_newMat = d_newConvMat;

	/*int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(d_prevMat, CONV_N * CONV_N * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(d_newMat, CONV_N * CONV_N * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(d_golKernel, 3 * 3 * sizeof(int), device, NULL);*/
	

	cudaFuncSetCacheConfig(kCalcGoLIteration, cudaFuncCachePreferL1);

	t1 = clock();
	for (int t = 0; t < nSteps; t++) {
		if (t % 1 == 0) printf("Iteracion #%d\n", t);
		kCalcGoLIteration << <dim3(nBlocks, nBlocks), dim3(nThreads, nThreads) >> > (d_prevMat, d_newMat, d_golKernel, n, CONV_N);
		cudaDeviceSynchronize();
		d_tempPtr = d_prevMat;
		d_prevMat = d_newMat;
		d_newMat = d_tempPtr;

		//cudaMemcpy(h_newConvMat, d_prevMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaMemcpy(h_newConvMat, d_prevMat, CONV_N * CONV_N * sizeof(int), cudaMemcpyDeviceToHost);
	t2 = clock();

	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			golFinalMat[j + i * n] = h_newConvMat[(j + 1) + (i + 1) * CONV_N];
		}
	}

	printf("Internal GPU Time: %f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	cudaFree(d_convMat);
	cudaFree(d_newConvMat);
	cudaFree(d_golKernel);
	free(h_newConvMat);
}