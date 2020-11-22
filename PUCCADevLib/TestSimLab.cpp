#include <stdio.h>
#include <string>
#include "CUDAExperiment.h"
#include "GoLGPUExperiment.cuh"
#include "GoLCPUExperiment.h"
#include "GoLCCA.cuh"
#include "mat_utils.h"
#include <typeinfo>



int main()
{
	printf("Hello SimLab!\n");

	//// Proof of Concept Experimets
	int amountOfVariablesPerCell = 1;
	int dimVarFactor = (int)sqrt(amountOfVariablesPerCell);
	GoLGPUExperiment golExperiment("GPU - Game of Life");
	printf("Initializing %s...\n", golExperiment.experimentName.c_str());
	golExperiment.setParamGridN(dimVarFactor*10);
	golExperiment.setParamStepsN(1);
	golExperiment.setParamThreads(16);
	golExperiment.setParamBlocks(128);
	golExperiment.initialize();
	golExperiment.runExperimentInstance();


	/*GoLCPUExperiment golCpuExperiment("CPU - Game of Life");
	printf("Initializing %s...\n", golCpuExperiment.experimentName.c_str());
	golCpuExperiment.setParamGridN(dimVarFactor * 1000);
	golCpuExperiment.setParamStepsN(1000);
	golCpuExperiment.initialize();
	golCpuExperiment.runExperimentInstance();

	golCpuExperiment.printResults();
	golExperiment.printResults();*/

	// Trying CA

	ExecutionEnv envs[] = { ENV_CPU, ENV_GPU };
	for (int i =0; i < 2; i++)
	{
		GoLCCA gol(4000);
		gol.setThreadsN(32);
		gol.init(envs[i]);

		printf("Beginning new GoLCA!\n");

		GoLGlobals global = gol.caGlobal;

		GoLCell* state = (GoLCell*)gol.getState();
		state[global.f2dTo1d(4, 4)].state = 1;
		state[global.f2dTo1d(4, 5)].state = 1;
		state[global.f2dTo1d(4, 6)].state = 1;

		gol.prepare();

		int* golMatState = gol.getStateIntMat();
		/*printf("Before State:\n");
		printMat(golMatState, gol.caGlobal.gridN);*/

		clock_t t1, t2;
		t1 = clock();		
		for (int i = 0; i < 20; i++) {
			gol.step();
			printf("Step (%d)\n", i);
			/*printf("After State:\n");
			golMatState = gol.getStateIntMat();
			printMat(golMatState, gol.caGlobal.gridN);*/
		}
		t2 = clock();
		float time = (double)(t2 - t1) / CLOCKS_PER_SEC;
		printf("Time: %lf\n", time);
	}

	printf("Finished Execution!\n");


	return 0;
}