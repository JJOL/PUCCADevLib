#pragma once
#include <string>
#include <stdio.h>
#include <time.h>
#include "CUDAExperiment.h"


class GoLGPUExperiment : public CUDAExperiment
{
private:
	int grid_n;
	int steps_n;
	int threads_n;
	int blocks_n;
	bool printEveryStep;

	int* h_golMat, * h_golFinalMat, * h_golKernel, * h_golExtMat;

	clock_t t1, t2;
public:
	GoLGPUExperiment(std::string name) : CUDAExperiment(name), grid_n(0), steps_n(0), threads_n(0), blocks_n(0), printEveryStep(0) {}
	void setPrintEveryStep(bool flag);
	void setParamGridN(int gridN);
	void setParamStepsN(int stepsN);
	void setParamThreads(int threadsN);
	void setParamBlocks(int blocksN);

	void initialize();
	void runExperimentInstance();
	void printResults();
};