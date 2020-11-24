#  Parallelized-Units CUDA Cellular Automata (PUCCA Dev Framework)
v0.3.0
---
A research and development framework for implementing ambicious parallel celullar automata on CUDA and integrating them into external prototypal tools or systems.

## Usage
The project contains a template file with guiding comments for programming a CUDA Cellular Automata and a sample of a Game of Life automaton.
It is possible to run your automaton as a stand alone CUDA C/C++ program and is suggested for initial validation and benchmarking. However, the source code is meant to be compiled into a shared library that can loaded by an external program, such as MASON as it is demonstrated.

For further information and a walkthrough of the development of the same Game of Life automaton, read the [report](PUCCA_A_proposed_framework_for_the_development_of_parallelized_cellula_automata.pdf).

## System Requirements
To build and run the source code contained in this repository it is required as a minimum:
1. A c/c++ compiler
2. CUDA Toolkit 5 or above
3. A NVIDIA Graphics Card with compute capanility above 20

You can benefit more if:
1. Your OS is Windows
2. You have Visual Studio with a well configured NVIDIA Toolkit Extension

## Sister Project
[JPUCCA](https://github.com/JJOL/JPUCCA/).

## License
[GNU General Public License 3.0](LICENSE)