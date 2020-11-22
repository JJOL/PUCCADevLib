#include "CUDACA.cuh"
#include <stdio.h>
#include <string.h>

void hostClean(void* address) { if (address != NULL) free(address); }
void deviceClean(void* address) { if (address != NULL) cudaFree(address); }

void cuCA::setState(void* stateData)
{
	memcpy(m_prevData, stateData, gridN * gridN * cellTSize);
}

void* cuCA::getState()
{
	return m_prevData;
}

void cuCA::init(ExecutionEnv _env)
{
	const int size = gridN * gridN;

	env = _env;

	cpuInit();

	if (env == ENV_GPU) {
		// Creating memory pointers for state on device
		cudaMalloc((void**)&dev_prevData, size * cellTSize);
		cudaMalloc((void**)&dev_nextData, size * cellTSize);

		cudaMemcpy(dev_prevData, m_prevData, size * cellTSize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_nextData, m_nextData, size * cellTSize, cudaMemcpyHostToDevice);

		gpuInit();
	}
}

void cuCA::prepare()
{
	int size = gridN * gridN;

	memcpy(m_nextData, m_prevData, gridN * gridN * cellTSize);

	if (env == ENV_GPU) {
		cudaMemcpy(dev_prevData, m_prevData, size * cellTSize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_nextData, m_nextData, size * cellTSize, cudaMemcpyHostToDevice);
	}
}

void cuCA::step()
{
	if (env == ENV_CPU)
		cpuStep();
	else if (env == ENV_GPU)
		gpuStep();
	else
		fprintf(stderr, "ERROR: No execution environment defined for cuCA!\n");
}

void cuCA::step(int nSteps)
{
	while (nSteps-- > 0) step();
}

void cuCA::setThreadsN(int _threadsN)
{
	threadsN = _threadsN;
}

void cuCA::cpuStep()
{
	void* temp;
	for (int y = 1; y < gridN - 1; y++) {
		for (int x = 1; x < gridN - 1; x++) {
			cpuCellCall(x, y, m_prevData, m_nextData);
		}
	}

	temp = m_prevData;
	m_prevData = m_nextData;
	m_nextData = temp;
}

void cuCA::gpuStep()
{
	int size = gridN * gridN;

	void* dev_temp;
	printf("Begin\n");
	gpuCellCall(dev_prevData, dev_nextData, size);
	//printf("End1\n");
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
	}
	
	//printf("End2\n");
	dev_temp = dev_prevData;
	dev_prevData = dev_nextData;
	dev_nextData = dev_temp;

	cudaMemcpy(m_prevData, dev_prevData, size * cellTSize, cudaMemcpyDeviceToHost);
}

cuCA::~cuCA()
{
	hostClean(m_prevData);
	hostClean(m_nextData);

	if (env == ENV_GPU) {
		deviceClean(dev_prevData);
		deviceClean(dev_nextData);
	}
}