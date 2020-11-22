#include "GoLGPUExperiment.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mat_utils.h"
#include "gpuGoLSim.cuh"

void GoLGPUExperiment::setParamGridN(int gridN) { grid_n = gridN; }
void GoLGPUExperiment::setParamStepsN(int stepsN) { steps_n = stepsN; }
void GoLGPUExperiment::setParamThreads(int threadsN) { threads_n = threadsN; }
void GoLGPUExperiment::setParamBlocks(int blocksN) { blocks_n = blocksN; }

void GoLGPUExperiment::initialize()
{
	int CONV_N = grid_n + 2;


	h_golMat = (int*)malloc(grid_n * grid_n * sizeof(int));
	h_golFinalMat = (int*)malloc(grid_n * grid_n * sizeof(int));
	h_golKernel = (int*)malloc(3 * 3 * sizeof(int));
	h_golExtMat = (int*)malloc(CONV_N * CONV_N * sizeof(int));

	if (h_golMat == NULL || h_golFinalMat == NULL || h_golKernel == NULL || h_golExtMat == NULL) {
		printf("ERROR: Could not create enough host memory!\n");
		exit(-1);
	}

	initMat(h_golMat, grid_n, 0);
	h_golMat[3 + 4 * grid_n] = 1;
	h_golMat[4 + 4 * grid_n] = 1;
	h_golMat[5 + 4 * grid_n] = 1;
	initMat(h_golFinalMat, grid_n, 0);
	initMat(h_golExtMat, CONV_N, 0);
	initMooreKernel(h_golKernel);

	printf("GPU Init Mat\n");
	printMat(h_golMat, grid_n);
}

void GoLGPUExperiment::runExperimentInstance()
{
	GoLCA::hCAInit(h_golExtMat, h_golKernel, h_golFinalMat, grid_n, blocks_n, threads_n);
	
	GoLCA::hCAReady();
	t1 = clock();
	for (int t = 0; t < steps_n; t++)
		GoLCA::hCAStep(t);
	GoLCA::hCADone();
	t2 = clock();

	printf("GPU Final Mat\n");
	printMat(h_golFinalMat, grid_n);

	free(h_golMat);
	free(h_golExtMat);
	free(h_golFinalMat);
	free(h_golKernel);
}

void GoLGPUExperiment::printResults()
{
	float gpuTime = (double)(t2 - t1) / CLOCKS_PER_SEC;

	printf("--------------\n");
	printf("Final GoL Mat:\n");
	printf("--------------\n");
	printf("N: %lu\n", grid_n);
	printf("N_STEPS: %d\n", steps_n);
	printf("N_BLOCKS: %dx%d=%d\n", blocks_n, blocks_n, blocks_n * blocks_n);
	printf("N_THREADS: %dx%d=%d\n", threads_n, threads_n, threads_n * threads_n);
	printf("GPU Time: %f\n", gpuTime);
}