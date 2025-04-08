# Optimizing Free-Form Grid Shells with Reclaimed Elements under Inventory Constraints

**Andrea Favilli<sup>a,b</sup>, Francesco Laccone<sup>a</sup>, Paolo Cignoni<sup>a</sup>, Luigi Malomo<sup>a</sup>, Daniela Giorgi<sup>a</sup>**  
<sup>a</sup>Institute of Information Science and Technologies "A. Faedo" (ISTI), National Research Council of Italy (CNR)  
<sup>b</sup>University of Pisa (Italy)


Paper: TBA \
BibTeX: TBA \
Website: TBA

![image](./images/teaser.svg)

# Installation
This code has been tested on Windows® 11 Pro and Ubuntu 24.04. The hardware setup is based on a Intel i9-14900KF CPU, 64 GB of RAM, NVIDIA GeForce RTX 4080 Super GPU with 16 GB of dedicated memory. The code runs on Python 3.11.5 with PyTorch 2.4.1, CUDA 11.8, PyTorch Geometric 2.6.1, Chamferdist 1.0.3, GurobiPy 11.0.3.

## Prerequisites
Gurobi Optimizer®, a platform for optimization and decision-making, is employed to solve Integer Linear Programming (ILP) instances inside the algorithm. A licence of Gurobi Optimizer®, free for [academic purposes](https://www.gurobi.com/academia/academic-program-and-licenses/?_gl=1*h5ziwn*_up*MQ..*_gs*MQ..*_ga*MjA2Mzg0Njc2Ny4xNzQ0MDM4NTc0*_ga_RTTPP25C8N*MTc0NDAzODU3My4xLjEuMTc0NDAzODU4Mi4wLjAuMTQ5NzE4MzcyMw..&gclid=Cj0KCQjw782_BhDjARIsABTv_JDVt-R_Rg3uSrZeey0R1Mxb2XZHQM-bNhYxDwC07DLZLR85LC1u0msaAgciEALw_wcB), is therefore needed. Once the user has got the licence, the corresponding .lic file has to be put in the home directory when installing software dependencies. \
Some of the implemented GPU-accelerated computations require the installation of the [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive). Please, check the correspondance of the installed CUDA Toolkit version with the version of the `pytorch-cuda` package (12.4 for both, in the case of the preconfigured environment we provide).


## Installing Anaconda dependencies
We employed [Anaconda](https://www.anaconda.com/products/distribution), a popular Python distribution for data science and machine learning. After that Anaconda is installed, we can use an Anaconda shell to create virtual environments and run the code. From an Anaconda prompt, we move to the repository root directory and enter the command
~~~
conda env create --file environment.yml
~~~
to create an envirorment named ```ReclaimedGridShells``` that contains all the needed dependencies. We can then activate ```ReclaimedGridShells``` by typing
~~~
conda activate ReclaimedGridShells
~~~
To ensure that CUDA Toolkit 12.4 works correctly, check for latest NVIDIA card driver update. 

## Compiling binaries
The replication of the paper figures requires the execution of compiled C++ code contained in the `cpp` folder. The user can decide to build the executables via `cpp\CMakeLists.txt` or to download pre-compiled binaries here (...).

# Usage