#include "GoLCCA.cuh"
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "mat_utils.h"

void hostClean(void* address) { if (address != NULL) free(address); }
void deviceClean(void* address) { if (address != NULL) cudaFree(address); }

GoLCCA::GoLCCA(int gridN) : m_gridN(gridN)
{
	env = ENV_UNSPECIFIED;
	caGlobal.gridN = gridN;
	// Init to NULL to be able to skip them during cleaning if we have no data
	caGlobal.neighborsKernel = NULL;
	m_nextData = NULL;
	dev_prevData = NULL;
	dev_nextData = NULL;
	dev_ca = NULL;
	m_kernel = NULL;
	m_prevData = NULL;
	m_data = NULL;
}

void GoLCCA::init(ExecutionEnv _env)
{
	env = _env;

	const int size = m_gridN * m_gridN;

	m_data = (int*)malloc(size * sizeof(int));
	initMat(m_data, m_gridN, 0);

	// Setup 
	int x, y;
	m_prevData = new GoLCell[size];
	for (x = 0; x < m_gridN; x++) {
		for (y = 0; y < m_gridN; y++) {
			m_prevData[caGlobal.f2dTo1d(x, y)].x = x;
			m_prevData[caGlobal.f2dTo1d(x, y)].y = y;
		}
	}
		
	m_nextData = new GoLCell[size];
	
	// Only GoL data
	caGlobal.neighborsKernel = (int*)malloc(3 * 3 * sizeof(int));
	initMooreKernel(caGlobal.neighborsKernel);

	if (env == ENV_GPU) {
		// Creating memory pointers for state on device
		cudaMalloc((void**)&dev_prevData, size * sizeof(GoLCell));
		cudaMalloc((void**)&dev_nextData, size * sizeof(GoLCell));

		cudaMemcpy(dev_prevData, m_prevData, size * sizeof(GoLCell), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_nextData, m_nextData, size * sizeof(GoLCell), cudaMemcpyHostToDevice);

		// Setting Up Common GoL device auxilary data
		GoLGlobals cpyGlobals;
		cpyGlobals.gridN = caGlobal.gridN;
		cudaMalloc((void**)&cpyGlobals.neighborsKernel, 3 * 3 * sizeof(int));
		cudaMalloc((void**)&dev_caGlobals, sizeof(GoLGlobals));
		cudaMemcpy(cpyGlobals.neighborsKernel, caGlobal.neighborsKernel, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_caGlobals, &cpyGlobals, sizeof(GoLGlobals), cudaMemcpyHostToDevice);
	}
}

GoLCCA::~GoLCCA()
{
	hostClean(caGlobal.neighborsKernel);
	hostClean(m_data);
	hostClean(m_prevData);
	hostClean(m_nextData);

	if (env == ENV_GPU) {
		deviceClean(dev_prevData);
		deviceClean(dev_nextData);
		GoLGlobals cpyDevGlobals;
		cudaMemcpy(&cpyDevGlobals, dev_caGlobals, sizeof(GoLGlobals), cudaMemcpyDeviceToHost);
		deviceClean(cpyDevGlobals.neighborsKernel);
		deviceClean(dev_caGlobals);
	}
}

void GoLCCA::setState(void* stateData)
{
	memcpy(m_prevData, stateData, m_gridN * m_gridN * sizeof(GoLCell));
}

void* GoLCCA::getState()
{
	return m_prevData;
}

void GoLCCA::prepare()
{
	int size = m_gridN * m_gridN;

	memcpy(m_nextData, m_prevData, m_gridN * m_gridN * sizeof(GoLCell));

	if (env == ENV_GPU) {
		cudaMemcpy(dev_prevData, m_prevData, size * sizeof(GoLCell), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_nextData, m_nextData, size * sizeof(GoLCell), cudaMemcpyHostToDevice);
	}
}

int* GoLCCA::getStateIntMat()
{
	for (int i = 0; i < m_gridN * m_gridN; i++)
		m_data[i] = m_prevData[i].state;
	return m_data;
}

void GoLCCA::cpuStep()
{
	GoLCell* temp;
	for (int y = 1; y < m_gridN - 1; y++) {
		for (int x = 1; x < m_gridN - 1; x++) {
			m_nextData[caGlobal.f2dTo1d(x, y)].update(&caGlobal, m_prevData);
		}
	}

	temp = m_prevData;
	m_prevData = m_nextData;
	m_nextData = temp;
}

__global__ void kGoLStep(GoLGlobals* ca, GoLCell* oldCAData, GoLCell* newCAData, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < n) {
		newCAData[tid].update(ca, oldCAData);
		tid += blockDim.x + gridDim.x;
	}

}

void GoLCCA::gpuStep()
{
	int size = m_gridN * m_gridN;

	GoLCell* dev_temp;

	if (dev_caGlobals == NULL) {
		fprintf(stderr, "ERROR: GoL dev_caGlobals is empty!\n");
		return;
	}
	kGoLStep <<<(size + size - 1) / 32, 32>> > (dev_caGlobals, dev_prevData, dev_nextData, size);
	cudaDeviceSynchronize();
	dev_temp = dev_prevData;
	dev_prevData = dev_nextData;
	dev_nextData = dev_temp;

	cudaMemcpy(m_prevData, dev_prevData, size * sizeof(GoLCell), cudaMemcpyDeviceToHost);
}



__host__ __device__ void GoLGlobals::f1dTo2d(int index, int* x, int* y)
{
	*x = index % gridN;
	*y = index / gridN;
}
 __forceinline__ __host__ __device__ int GoLGlobals::f2dTo1d(int x, int y)
{
	return x + y * gridN;
}



/* GoL Closest Logic at Cell Level */
__host__ __device__ void GoLCell::update(GoLGlobals* ca, GoLCell* prevCAData)
{
	int aliveNeighbors = 0;

	for (int k_row = 0; k_row < 3; k_row++) {
		for (int k_col = 0; k_col < 3; k_col++) {
			aliveNeighbors +=
				ca->neighborsKernel
				[k_col + (k_row * 3)] * prevCAData[ca->f2dTo1d(x - 1 + k_col, y - 1 + k_row)].state;
		}
	}

	state = (1 - state) * (aliveNeighbors == 3) + (state) * (aliveNeighbors == 2 || aliveNeighbors == 3);
}