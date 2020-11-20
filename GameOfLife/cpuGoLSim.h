#include <stdlib.h>
#include <stdio.h>
#include "mat_utils.h"

void golConvolution(int row, int col, int* prevMat, int* newMat, int* ker, int n, int convN)
{
	int index = (col + 1) + (row + 1) * convN;

	int aliveNeighbors = 0;

	for (int ki = 0; ki < 3; ki++) {
		for (int kj = 0; kj < 3; kj++) {
			aliveNeighbors +=
				ker[kj + ki * 3] * prevMat[(col + kj) + (row + ki) * convN];
		}
	}
	
	int cellValue = prevMat[index];

	if (cellValue == 0) {
		if (aliveNeighbors == 3) cellValue = 1;
		else cellValue = 0;
	}
	else {
		if (aliveNeighbors == 2 || aliveNeighbors == 3) cellValue = 1;
		else cellValue = 0;
	}

	newMat[index] = cellValue;
}

void cpuPlayGoL(int* initGolMat, int* golKernel, int* golFinalMat, int n, int nSteps)
{
	int CONV_N = n + 2;
	int* cpyConvMat;
	cpyConvMat = (int*)malloc(CONV_N * CONV_N * sizeof(int));
	initMat(cpyConvMat, CONV_N, 0);



	int* prevMat, * newMat, * tempPtr;
	prevMat = initGolMat;
	newMat = cpyConvMat;

	for (int t = 0; t < nSteps; t++) {

		if (t % 1 == 0) printf("Iteracion #%d\n", t);
		/*printMat(prevMat, CONV_N);*/

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {


				golConvolution(j, i, prevMat, newMat, golKernel, n, CONV_N);
			}
		}

		// Swap Convolution Matrices to reutilize
		tempPtr = prevMat;
		prevMat = newMat;
		newMat = tempPtr;
	}

	// Copy Results into extendend conv mat
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			golFinalMat[j + i * n] = prevMat[(j + 1) + (i + 1) * CONV_N];
		}
	}

	free(cpyConvMat);
}