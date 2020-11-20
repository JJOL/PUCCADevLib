#pragma once
#include <string>
#include <stdio.h>
#include <time.h>
#include "CUDAExperiment.h"

class GoLCPUExperiment : public CUDAExperiment
{
private:
	int grid_n;
	int steps_n;

	int* h_golMat, * h_golFinalMat, * h_golKernel, * h_golExtMat;

	clock_t t1, t2;
public:
	GoLCPUExperiment(std::string name) : CUDAExperiment(name), grid_n(0), steps_n(0) {}
	void setParamGridN(int gridN) { grid_n = gridN; }
	void setParamStepsN(int stepsN) { steps_n = stepsN; }

	void initialize();
	void runExperimentInstance();
	void printResults();
};