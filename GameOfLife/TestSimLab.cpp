#include <stdio.h>
#include <string>
#include "CUDAExperiment.h"
#include "GoLGPUExperiment.cuh"
#include "GoLCPUExperiment.h"


int main()
{
	printf("Hello SimLab!\n");

	GoLGPUExperiment golExperiment("GPU - 6Game of Life");
	printf("Initializing %s...\n", golExperiment.experimentName.c_str());
	golExperiment.setParamGridN(1000);
	golExperiment.setParamStepsN(1000);
	golExperiment.setParamThreads(16);
	golExperiment.setParamBlocks(128);
	golExperiment.initialize();
	golExperiment.runExperimentInstance();


	GoLCPUExperiment golCpuExperiment("CPU - Game of Life");
	printf("Initializing %s...\n", golCpuExperiment.experimentName.c_str());
	golCpuExperiment.setParamGridN(1000);
	golCpuExperiment.setParamStepsN(1000);
	golCpuExperiment.initialize();
	golCpuExperiment.runExperimentInstance();

	golCpuExperiment.printResults();
	golExperiment.printResults();

	return 0;
}