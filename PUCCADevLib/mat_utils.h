/*
* Author: Juan Jose Olivera Loyola
* Name: PUCCA mat_utils.h
* Description:
* A utility library for performing matrix operations
*/
#pragma once
#include <stdio.h>

namespace PUCCA
{
	/*
	* initMat
	* Assigns a given value to the entries of the matrix
	* @Params
	*	int* mat: Memory address of the matrix
	*	int n: Side dimension of the matrix
	*	int val: Value given to each entry
	*/
	void initMat(int* mat, int n, int val);
	/*
	* initMooreKernel
	* Assigns the Moore Neighborhood Kernel values to the matrix
	* @Params
	*	int* kernel: Memory address of the kernel matrix
	*/
	void initMooreKernel(int* kernel);
	/*
	* copyMatIntoMat
	* Copies a square section data from srcMat into destMat
	* @Params
	*	int* srcMat: Matrix from where to copy data
	* 	int* destMat: Matrix to where to paste data
	*	int srcN: Side dimension of the matrix srcMat
	*	int destN: Side dimension of the matrix destMat
	*	int srcXoff: X offset of rectangle from where to start copying data in srcMat
	*	int srcYoff: Y offset of rectangle from where to start copying data in srcMat
	*	int destXoff: X offset of rectangle from where to start pasting data in destMat
	*	int destYoff: Y offset of rectangle from where to start pasting data in destMat
	*/
	void copyMatIntoMat(int* srcMat, int* destMat, int srcN, int destN, int srcXoff, int srcYoff, int destXoff, int destYoff);
	/*
	* initMat
	* Prints the entries of a matrix
	* @Params
	*	int* mat: Memory address of the matrix
	*	int n: Side dimension of the matrix
	*/
	void printMat(int* mat, int n);
	/*
	* 2dTo1d
	* Computes a 1 dimensional index from 2 dimensions (x and y)
	* It has 2 implementatios, one for host code and the other for device code
	* @Params
	*	int x
	*	int y
	*	int n
	* @Returns index
	*/
	int h2dTo1d(int x, int y, int n);
}

