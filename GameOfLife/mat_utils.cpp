#include "mat_utils.h"
#include <stdio.h>

void initMat(int* mat, int n, int val)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			mat[j + i * n] = val;
		}
	}
}

void initMooreKernel(int* kernel)
{
	initMat(kernel, 3, 1);
	kernel[1 + 1 * 3] = 0;
}


void copyMatIntoMat(int* srcMat, int* destMat, int srcN, int destN, int xoff, int yoff)
{
	for (int i = 0; i < srcN; i++) {
		for (int j = 0; j < srcN; j++) {
			destMat[(xoff + j) + (yoff + i) * destN] = srcMat[j + i * srcN];
		}
	}
}

void printMat(int* mat, int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d\t", mat[j + i * n]);
		}
		printf("\n");
	}
	printf("\n");
}
