#pragma once
#include "CUDACA.cuh"
#include "cuda_runtime.h"

class GoLGlobals : public cuCAGlobals
{
public:
	int* neighborsKernel;
};

class GoLCell : public cuCACell
{
public:
	//int x, y;
	int state;
	GoLCell(): cuCACell(), state(0) {}
	PARALLEL void update(GoLGlobals* ca, GoLCell* prevCAData);
};

class GoLCCA : public cuCA
{
private:
	GoLGlobals* dev_caGlobals;
	int *m_data;
public:
	GoLGlobals caGlobal;
public:
	GoLCCA(int gridN);
	~GoLCCA();
	int* getStateIntMat();
private:
	void dataInit(int size);
	void cpuInit();
	void gpuInit();
	void cpuCellCall(int x, int y, void* prevData, void* nextData);
	void gpuCellCall(void* dev_prevData, void* dev_nextData, int size);
};