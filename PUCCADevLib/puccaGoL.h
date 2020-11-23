#pragma once
/*
* module: GoLCA.h
* Define the CA public API
*/

namespace GoLCA {
	void hCAInit(int* _initGolMat, int* _golKernel, int* _golFinalMat, int _n, int _nBlocks, int _nThreads);
	void hCAReady();
	void hCAStep();
	void hCARead();
	void hCADone();
}