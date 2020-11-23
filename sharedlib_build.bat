Rem This is the compilation process of PUCCA CA implementation of MASON Java JNI

Rem Execute it Visual Studio Developer Prompt Mode 64bits
Rem For activating 64 bits mode run equivalent:
Rem "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64


mkdir objs

IF NOT EXIST bin\NUL mkdir bin

Rem Compile CA Implementation
nvcc -c puccaGoL.cu -o objs\puccaGoL.obj

Rem Compile Helper PUCCA C classes
cl -c mat_utils.cpp /Foobjs\mat_utils.obj

Rem This is for compiling Java JNI bridge for PUCCA
cl -I "C:\Program Files\Java\jdk1.8.0_261\include" -I "C:\Program Files\Java\jdk1.8.0_261\include\win32" -c sim_app_jpuccagol_JPGoLCA.cpp /Foobjs\sim_app_jpuccagol_JPGoLCA.obj

Rem Link the objects into final shared library
nvcc objs\sim_app_jpuccagol_JPGoLCA.obj objs\puccaGoL.obj objs\mat_utils.obj --shared -o bin\sim_app_jpuccagol_JPGoLCA.dll

rmdir objs