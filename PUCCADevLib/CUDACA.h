#pragma once
typedef enum ExecutionEnv {
	ENV_UNSPECIFIED,
	ENV_CPU,
	ENV_GPU
};

class cuCA
{
protected:
	ExecutionEnv env;
	int threadsN;
public:
	virtual void init(ExecutionEnv _env) = 0;
	virtual void setState(void* stateData) = 0;
	virtual void* getState() = 0;
	void step();
	void step(int nSteps);
	// CUDA Configuration
	void setThreadsN(int _threadsN);
private:
	virtual void cpuStep() = 0;
	virtual void gpuStep() = 0;
};

