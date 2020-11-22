#include "CloudSimCA.cuh"

#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include "mat_utils.h"

CSCCA::CSCCA(int gridN) : cuCA(gridN)
{
	caGlbs.gridN = gridN;

	cellTSize = sizeof(CSCell);
	globTSize = sizeof(CSGlobals);
}

void CSCCA::dataInit(int size)
{
	// Setup 
	int x, y;
	m_prevData = new CSCell[size];
	for (x = 0; x < caGlbs.gridN; x++) {
		for (y = 0; y < caGlbs.gridN; y++) {
			((CSCell*)m_prevData)[caGlbs.f2dTo1d(x, y)].x = x;
			((CSCell*)m_prevData)[caGlbs.f2dTo1d(x, y)].y = y;
		}
	}

	m_nextData = new CSCell[size];
}

void CSCCA::cpuInit()
{
	const int size = caGlbs.gridN * caGlbs.gridN;
	dataInit(size);
}

void CSCCA::gpuInit()
{
	// Setting Up Common GoL device auxilary data
	CSGlobals cpyGlobals;

	cudaMalloc((void**)&dev_caGlbs, globTSize);
	cudaMemcpy(dev_caGlbs, &cpyGlobals, globTSize, cudaMemcpyHostToDevice);
}

CSCCA::~CSCCA()
{
	if (env == ENV_GPU) {
		deviceClean(dev_caGlbs);
	}
}


void CSCCA::cpuCellCall(int x, int y, void* prevData, void* nextData)
{
	CSCell* golPrevData = (CSCell*)prevData;
	CSCell* golNextData = (CSCell*)nextData;

	golNextData[caGlbs.f2dTo1d(x, y)].update(&caGlbs, golPrevData);
}

__global__ void kCSStep(CSGlobals* ca, void* oldCAData, void* newCAData, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n) {
		((CSCell*)newCAData)[tid].update(ca, (CSCell*)oldCAData);
		tid += blockDim.x + gridDim.x;
	}
}
void CSCCA::gpuCellCall(void* dev_prevData, void* dev_nextData, int size)
{
	kCSStep << <(size + size - 1) / threadsN, threadsN >> > (dev_caGlbs, dev_prevData, dev_nextData, size);
}

PARALLEL void CSCell::update(CSGlobals* ca, CSCell* prevCAData)
{

}