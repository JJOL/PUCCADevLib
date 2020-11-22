#pragma once
#include "CUDACA.h"
#include "cuda_runtime.h"

class GoLCCA;
class GoLGlobals;
class GoLCell;

class GoLCCA : public cuCA
{
private:
	GoLCell* m_nextData;
	GoLCell* dev_prevData;
	GoLCell* dev_nextData;
	GoLCCA* dev_ca;
	GoLCell* m_prevData;
	int *m_data;
public:
	const int m_gridN;
	int* m_kernel;

public:
	GoLCCA(int gridN);
	~GoLCCA();

	void setState(void* stateData);
	void* getState();

	void init(ExecutionEnv _env);
	void prepare();

	int* getStateIntMat();

	__host__ __device__ void f1dTo2d(int index, int *x, int *y);
	__host__ __device__ int f2dTo1d(int x, int y);

private:
	void cpuStep();
	void gpuStep();
};

//class GoLGlobals
//{
//public:
//	int* neighborsKernel;
//	int  gridN;
//	__host__ __device__ void f1dTo2d(int index, int* x, int* y);
//	__host__ __device__ int f2dTo1d(int x, int y);
//};


class GoLCell
{
public:
	int x, y;
	int state;
	GoLCell(): x(0), y(0), state(0) {}
	GoLCell(int _x, int _y) : x(_x), y(_y), state(0) {}
	__host__ __device__ void update(GoLCCA* ca, GoLCell* prevCAData);
};