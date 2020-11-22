#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>
#define PARALLEL __host__ __device__

void hostClean(void* address);
void deviceClean(void* address);

typedef enum ExecutionEnv {
	ENV_UNSPECIFIED,
	ENV_CPU,
	ENV_GPU
} ExecutionEnv;

class cuCAGlobals
{
public:
	int gridN;
	__forceinline__ __host__ __device__ int f2dTo1d(int x, int y) const { return x + y * gridN; }
};

class cuCACell
{
public:
	int x, y;
	cuCACell() : x(0), y(0) {}
};

class cuCA
{
private:
	int gridN;
protected:
	ExecutionEnv env;
	int threadsN;
	int cellTSize;
	int globTSize;

	void* m_prevData;
	void* m_nextData;

	void* dev_prevData;
	void* dev_nextData;
public:
public:
	cuCA(int _gridN) : gridN(_gridN), env(ENV_UNSPECIFIED),
		m_prevData(NULL), m_nextData(NULL), dev_prevData(NULL), dev_nextData(NULL) {};
	virtual ~cuCA();
	void init(ExecutionEnv _env);
	void setState(void* stateData);
	void* getState();
	void prepare();
	void step();
	void step(int nSteps);
	void cpuStep();
	void gpuStep();
	// CUDA Configuration
	void setThreadsN(int _threadsN);
private:
	virtual void cpuInit() = 0;
	virtual void gpuInit() = 0;
	virtual void cpuCellCall(int x, int y, void* prevData, void* nextData) = 0;
	virtual void gpuCellCall(void* dev_prevData, void* dev_nextData, int size) = 0;
};

