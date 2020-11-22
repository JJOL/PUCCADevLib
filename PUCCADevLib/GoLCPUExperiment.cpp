#include "GoLCPUExperiment.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mat_utils.h"
#include "cpuGoLSim.h"

void GoLCPUExperiment::initialize()
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
}

void GoLCPUExperiment::runExperimentInstance()
{
	int CONV_N = grid_n + 2;

	t1 = clock();
	copyMatIntoMat(h_golMat, h_golExtMat, grid_n, CONV_N, 0, 0, 1, 1);
	cpuPlayGoL(h_golExtMat, h_golKernel, h_golFinalMat, grid_n, steps_n);
	//printMat(h_golFinalMat, N);
	t2 = clock();
}

void GoLCPUExperiment::printResults()
{
	float cpuTime = (double)(t2 - t1) / CLOCKS_PER_SEC;

	printf("--------------\n");
	printf("Final GoL Mat:\n");
	printf("--------------\n");
	printf("N: %lu\n", grid_n);
	printf("N_STEPS: %d\n", steps_n);
	printf("CPU Time: %f\n", cpuTime);
}