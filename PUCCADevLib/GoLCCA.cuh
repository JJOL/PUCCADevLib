#pragma once
#include "CUDACA.h"
#include "cuda_runtime.h"

class GoLCCA;
class GoLGlobals;
class GoLCell;

class GoLGlobals
{
public:
	int* neighborsKernel;
	int  gridN;
	__host__ __device__ void f1dTo2d(int index, int* x, int* y);
	__forceinline__ __host__ __device__ int f2dTo1d(int x, int y);
};

class GoLCell
{
public:
	int x, y;
	int state;
	GoLCell() : x(0), y(0), state(0) {}
	GoLCell(int _x, int _y) : x(_x), y(_y), state(0) {}
	__host__ __device__ void update(GoLGlobals* ca, GoLCell* prevCAData);
};

class GoLCCA : public cuCA
{
private:
	GoLCell* m_nextData;
	GoLCell* dev_prevData;
	GoLCell* dev_nextData;
	GoLCCA* dev_ca;
	GoLGlobals* dev_caGlobals;
	GoLCell* m_prevData;
	int *m_data;
public:
	const int m_gridN;
	int* m_kernel;
	GoLGlobals caGlobal;
public:
	GoLCCA(int gridN);
	~GoLCCA();

	void init(ExecutionEnv _env);
	void prepare();

	void setState(void* stateData);
	void* getState();
	int* getStateIntMat();
private:
	void cpuStep();
	void gpuStep();
};