/*
* Author: Juan Jose Olivera Loyola
* Name: PUCCA mat_utils.cpp
* Description:
* Implementation of functions defined at mat_utils.h
*/
#include "mat_utils.h"
#include <stdio.h>

namespace PUCCA
{

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

void copyMatIntoMat(int* srcMat, int* destMat, int srcN, int destN, int srcXoff, int srcYoff, int destXoff, int destYoff)
{
	for (int i = 0; i < srcN; i++) {
		for (int j = 0; j < srcN; j++) {
			destMat[(destXoff + j) + (destYoff + i) * destN] = srcMat[j + i * srcN];
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

int h2dTo1d(int x, int y, int m)
{
	return x + y * m;
}

__device__ int k2dTo1d(int x, int y, int m)
{
	return x + y * m;
}

}
