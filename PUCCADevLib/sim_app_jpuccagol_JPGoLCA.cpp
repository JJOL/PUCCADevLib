#include "sim_app_jpuccagol_JPGoLCA.h"
#include "mat_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include "puccaGoL.h"

/*
* Data Variables to hold input and out CA state
*/
int* ext_ca_in_data, * ca_out_data;

/*
* Constant Variables that shoould not change
*/
int* kernel;
int gridN, EXT_N;



JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_bind
(JNIEnv* env, jobject javaobject, jintArray jiarr, jint n)
{
	kernel = (int*)malloc(3 * 3 * sizeof(int));
	PUCCA::initMooreKernel(kernel);


	jint* app_ca_data = env->GetIntArrayElements(jiarr, NULL);
	int* cApp_ca_data = (int*)app_ca_data;
	gridN = n;

	EXT_N = gridN + 2;
	ext_ca_in_data = (int*)malloc(EXT_N * EXT_N * sizeof(int));
	ca_out_data = (int*)malloc(gridN * gridN * sizeof(int));
	PUCCA::initMat(ext_ca_in_data, EXT_N, 0);
	PUCCA::initMat(ca_out_data, gridN, 0);

	PUCCA::copyMatIntoMat(cApp_ca_data, ext_ca_in_data, gridN, EXT_N, 0, 0, 1, 1);

	GoLCA::hCAInit(ext_ca_in_data, kernel, ca_out_data, gridN, 128, 16);

	env->ReleaseIntArrayElements(jiarr, app_ca_data, 0);

	return true;
}

JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_ready
(JNIEnv* env, jobject javaobject)
{
	GoLCA::hCAReady();
	return true;
}

JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_step
(JNIEnv* env, jobject javaobject)
{
	GoLCA::hCAStep();
	return true;
}

JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_write
(JNIEnv* env, jobject javaobject, jintArray jiarr, jint n)
{
	jint* app_ca_data = env->GetIntArrayElements(jiarr, NULL);
	int* cApp_ca_data = (int*)app_ca_data;

	if (n == gridN) {
		PUCCA::copyMatIntoMat(cApp_ca_data, ext_ca_in_data, gridN, EXT_N, 0, 0, 1, 1);
		GoLCA::hCAReady();
	}
	else {
		env->ReleaseIntArrayElements(jiarr, app_ca_data, 0);
		return false;
	}
	env->ReleaseIntArrayElements(jiarr, app_ca_data, 0);
	return true;
}

JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_read
(JNIEnv* env, jobject javaobject, jintArray jiarr, jint n)
{
	jint len = env->GetArrayLength(jiarr);

	GoLCA::hCARead();
	env->SetIntArrayRegion(jiarr, 0, gridN * gridN, (jint*)ca_out_data);

	return true;
}

JNIEXPORT jboolean JNICALL Java_sim_app_jpuccagol_JPGoLCA_done
(JNIEnv* env, jobject javaobject)
{
	GoLCA::hCADone();

	if (ext_ca_in_data != NULL) {
		free(ext_ca_in_data);
	}
	if (ca_out_data != NULL) {
		free(ca_out_data);
	}
	if (kernel != NULL) {
		free(kernel);
	}

	return true;
}