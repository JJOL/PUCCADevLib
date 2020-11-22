#pragma once

#include "CUDACA.cuh"

class CSGlobals : public cuCAGlobals
{
public:
	int tempD;
	int P;
};

class CSCell
{
public:
	int gX, gY;
	float x, y, dx, dy;
	float ux, uy;
	float t;
	float p0;
	CSCell() : gX(0), gY(0), x(0), y(0), dx(0), dy(0), ux(0), uy(0), t(0), p0(0) {}
	PARALLEL void update(CSGlobals* glbs, CSCell* prevStateData);
};

class CSCCA : public cuCA
{
private:
	CSGlobals* dev_caGlbs;
public:
	CSGlobals caGlbs;
public:
	CSCCA(int gridN);
	~CSCCA();
	int* getStateIntMat();
private:
	void dataInit(int size);
	void cpuInit();
	void gpuInit();
	void cpuCellCall(int x, int y, void* prevData, void* nextData);
	void gpuCellCall(void* dev_prevData, void* dev_nextData, int size);
};