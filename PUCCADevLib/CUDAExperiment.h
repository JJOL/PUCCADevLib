#pragma once
#include <string>

class CUDAExperiment
{
private:
public:
	std::string experimentName;
	CUDAExperiment(std::string _experimentName) : experimentName(_experimentName) {}

	virtual void initialize() = 0;
	virtual void repeat(int n) {};
	virtual void printResults() = 0;
};
