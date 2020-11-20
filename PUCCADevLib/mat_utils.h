#pragma once

#include <stdio.h>


void initMat(int* mat, int n, int val);

void initMooreKernel(int* kernel);

void copyMatIntoMat(int* srcMat, int* destMat, int srcN, int destN, int xoff, int yoff);

void printMat(int* mat, int n);
