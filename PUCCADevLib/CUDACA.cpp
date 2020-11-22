#include "CUDACA.h"
#include <stdio.h>

void cuCA::step()
{
	if (env == ENV_CPU)
		cpuStep();
	else if (env == ENV_GPU)
		gpuStep();
	else
		fprintf(stderr, "ERROR: No execution environment defined for cuCA!\n");
		// throw exception();
}

void cuCA::step(int nSteps)
{
	while (nSteps-- > 0) step();
}

void cuCA::setThreadsN(int _threadsN)
{
	threadsN = _threadsN;
}