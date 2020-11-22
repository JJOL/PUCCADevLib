#include "GoLCCA.cuh"
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "mat_utils.h"

GoLCCA::GoLCCA(int gridN) : cuCA(gridN)
{
	caGlobal.gridN = gridN;
	caGlobal.neighborsKernel = NULL;
	m_data = NULL;

	cellTSize = sizeof(GoLCell);
	globTSize = sizeof(GoLGlobals);
	printf("Cell Size: %d\n", cellTSize);
}

void GoLCCA::dataInit(int size)
{
	// Setup 
	int x, y;
	m_prevData = new GoLCell[size];
	for (x = 0; x < caGlobal.gridN; x++) {
		for (y = 0; y < caGlobal.gridN; y++) {
			((GoLCell*)m_prevData)[caGlobal.f2dTo1d(x, y)].x = x;
			((GoLCell*)m_prevData)[caGlobal.f2dTo1d(x, y)].y = y;
		}
	}

	m_nextData = new GoLCell[size];
}

void GoLCCA::cpuInit()
{
	const int size = caGlobal.gridN * caGlobal.gridN;
	dataInit(size);
	
	caGlobal.neighborsKernel = (int*)malloc(3 * 3 * sizeof(int));
	initMooreKernel(caGlobal.neighborsKernel);

	m_data = (int*)malloc(size * sizeof(int));
	initMat(m_data, caGlobal.gridN, 0);
}

void GoLCCA::gpuInit()
{
	// Setting Up Common GoL device auxilary data
	GoLGlobals cpyGlobals;
	cpyGlobals.gridN = caGlobal.gridN;
	cudaMalloc((void**)&cpyGlobals.neighborsKernel, 3 * 3 * sizeof(int));
	cudaMalloc((void**)&dev_caGlobals, globTSize);
	cudaMemcpy(cpyGlobals.neighborsKernel, caGlobal.neighborsKernel, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_caGlobals, &cpyGlobals, globTSize, cudaMemcpyHostToDevice);
}

GoLCCA::~GoLCCA()
{
	hostClean(caGlobal.neighborsKernel);
	hostClean(m_data);

	if (env == ENV_GPU) {
		GoLGlobals cpyDevGlobals;
		cudaMemcpy(&cpyDevGlobals, dev_caGlobals, sizeof(GoLGlobals), cudaMemcpyDeviceToHost);
		deviceClean(cpyDevGlobals.neighborsKernel);
		deviceClean(dev_caGlobals);
	}
}

int* GoLCCA::getStateIntMat()
{
	for (int i = 0; i < caGlobal.gridN * caGlobal.gridN; i++)
		m_data[i] = ((GoLCell*)m_prevData)[i].state;
	return m_data;
}


void GoLCCA::cpuCellCall(int x, int y, void* prevData, void* nextData)
{
	GoLCell* golPrevData = (GoLCell*)prevData;
	GoLCell* golNextData = (GoLCell*)nextData;

	golNextData[caGlobal.f2dTo1d(x, y)].update(&caGlobal, golPrevData);
}

__device__ void update(int x, int y, int w, GoLGlobals* ca, GoLCell* prevCAData, GoLCell* nextCAData)
{
	int state = 0;
	//int state =  prevCAData[x].state;
	int aliveNeighbors = 0;

	int k_row, k_col;
	for (k_row = 0; k_row < 3; k_row++) {
		for (k_col = 0; k_col < 3; k_col++) {
			aliveNeighbors += 1;
			//ca->neighborsKernel
			//[k_col + (k_row * 3)] * prevCAData[x].state; //* prevCAData[ca->f2dTo1d(x - 1 + k_col, y - 1 + k_row)].state;
		}
	}

	state = (1 - state) * (aliveNeighbors == 3) + (state) * (aliveNeighbors == 2 || aliveNeighbors == 3);
	//state = aliveNeighbors;
	//nextCAData[x].state = state;
}

__global__ void kGoLStep(GoLGlobals* ca, GoLCell* oldCAData, GoLCell* newCAData, int n, int m)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n) {
		//((GoLCell*)newCAData)[tid].update(ca, (GoLCell*)oldCAData);
		//newCAData[tid].update(ca, NULL);
		update(tid, 1, m, ca, oldCAData, newCAData);
		tid += blockDim.x + gridDim.x;
	}
}
void GoLCCA::gpuCellCall(void* dev_prevData, void* dev_nextData, int size)
{
	printf("Size: %d\n", size);
	kGoLStep <<<(size + size - 1) / threadsN, threadsN >>> (dev_caGlobals, (GoLCell*)dev_prevData, (GoLCell*)dev_nextData, size, caGlobal.gridN);
}


/* GoL Closest Logic at Cell Level */



PARALLEL void GoLCell::update(GoLGlobals* ca, GoLCell* prevCAData)
{
	int aliveNeighbors = 0;

	for (int k_row = 0; k_row < 3; k_row++) {
		for (int k_col = 0; k_col < 3; k_col++) {
			aliveNeighbors += 1;
				/*ca->neighborsKernel
				[k_col + (k_row * 3)] * prevCAData[ca->f2dTo1d(x - 1 + k_col, y - 1 + k_row)].state;*/
		}
	}

	state = (1 - state) * (aliveNeighbors == 3) + (state) * (aliveNeighbors == 2 || aliveNeighbors == 3);
	state = 0;
}