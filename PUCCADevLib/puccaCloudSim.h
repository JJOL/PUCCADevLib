#pragma once
/*
* module: CloudCA.h
* Define the CA public API
*/

namespace CloudCA {
	void hCAInit(int* _initGolMat, int* _golFinalMat, int _n, int _nBlocks, int _nThreads);
	void hCAReady();
	void hCAStep();
	void hCARead();
	void hCADone();
}