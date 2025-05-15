# Optimizing Free-Form Grid Shells with Reclaimed Elements under Inventory Constraints

**Andrea Favilli<sup>a,b</sup>, Francesco Laccone<sup>a</sup>, Paolo Cignoni<sup>a</sup>, Luigi Malomo<sup>a</sup>, Daniela Giorgi<sup>a</sup>**  
<sup>a</sup>Institute of Information Science and Technologies "A. Faedo" (ISTI), National Research Council of Italy (CNR)  
<sup>b</sup>University of Pisa (Italy)


Paper: [link](https://vcgdata.isti.cnr.it/Publications/2025/EG-ReclaimedGridShells/Optimizing%20Free-Form%20Grid%20Shells%20with%20Reclaimed%20Elements.pdf) \
Website: [link](https://vcg.isti.cnr.it/publication/2025/ReclaimedGridShells/)

<img src="./images/teaser.svg" alt="Teaser" style="width:1200px; height:auto;">

# Installation
This code has been tested on Windows® 11 Pro and Ubuntu 24.04. The hardware setup is based on a Intel i9-14900KF CPU, 64 GB of RAM, NVIDIA GeForce RTX 4080 Super GPU with 16 GB of dedicated memory. The code runs on Python 3.11.5 with PyTorch 2.5.0 (2.4.1 for Ubuntu), CUDA 12.4 (12.1 for Ubuntu), PyTorch Geometric, GurobiPy 11.0.3. <ins>The use of CUDA is recommended to have results in reasonable computation time.</ins>

## Prerequisites
Gurobi Optimizer®, a platform for optimization and decision-making, is employed to solve Integer Linear Programming (ILP) instances inside the algorithm. A licence of Gurobi Optimizer®, free for [academic purposes](https://www.gurobi.com/academia/academic-program-and-licenses/?_gl=1*h5ziwn*_up*MQ..*_gs*MQ..*_ga*MjA2Mzg0Njc2Ny4xNzQ0MDM4NTc0*_ga_RTTPP25C8N*MTc0NDAzODU3My4xLjEuMTc0NDAzODU4Mi4wLjAuMTQ5NzE4MzcyMw..&gclid=Cj0KCQjw782_BhDjARIsABTv_JDVt-R_Rg3uSrZeey0R1Mxb2XZHQM-bNhYxDwC07DLZLR85LC1u0msaAgciEALw_wcB), is therefore needed. Once the user has got the licence, the corresponding .lic file has to be put in the home directory when installing software dependencies. \
Some of the implemented GPU-accelerated computations require the installation of the CUDA Toolkit ([link Win](https://developer.nvidia.com/cuda-12-4-0-download-archive), [link Ubuntu](https://developer.nvidia.com/cuda-12-1-0-download-archive)). Please, check the correspondance of the installed CUDA Toolkit version with the version of the `pytorch-cuda` package (12.4, Windows®, or 12.1, Ubuntu, for both, in the case of the preconfigured environments we provide).


## Installing Anaconda dependencies
We employed [Anaconda](https://www.anaconda.com/products/distribution), a popular Python distribution for data science and machine learning. After that Anaconda is installed, we can use an Anaconda shell to create virtual environments and run the code. From an Anaconda prompt, we move to the repository root directory and enter the command
~~~
conda env create --file environment_<win,ubu>.yml
~~~
to create an envirorment named ```ReclaimedGridShells``` that contains all the needed dependencies. We can then activate ```ReclaimedGridShells``` by typing
~~~
conda activate ReclaimedGridShells
~~~
To ensure that CUDA Toolkit works correctly, check for latest NVIDIA card driver update. 

## Compiling binaries
The replication of the paper figures requires the execution of compiled C++ code contained in the `cpp` folder. The user can decide to build the executables via `cpp\CMakeLists.txt` or to download pre-compiled binaries here (...).

# Usage
We present three use scenarios of this software, giving each time a figure from the orginal paper as application example. The given output folder will contain the 3D models for both inputs and outputs, together with the images with the same visualization style of the paper and some csv files containing quantitative results.

## Single case optimization: Figure 11
This task solves the optimization of an input gridshell shape, provided as a `.ply` file with fixed mesh topology, both in terms of static compliance reduction and reuse waste from a given stock of beam elements. The user can choose between one of the pre-configured
stocks `uniform`, `nonuniform1`, `nonuniform2`.

More specifically, the command
~~~
python single_task_exec.py --device "cuda" --meshpath "meshes/Dolphin.ply" -niter 10000 --lr 5e-6 --saveinterval 100 --times --stock "uniform" --curves --hists --structhist --rendermodels --render --renderw --jpg
~~~
performs single case computation of the shape `Dolphin.ply` and reproduces the images of Fig. 11. In the `output\Dolphin_stock_uniform` subfolder, the user can find the input and output gridshell (files `Dolphin_start.jpg` and `Dolphin_end.jpg`, respectively), the color-coded wireframes showing the initial and final beam assignment (files `Dolphin_batch_colors_start.jpg` and `Dolphin_batch_colors_end.jpg`), the mesh in false colors representing the initial and final Service Load displacements (files `Dolphin_deflections_start.jpg`, `Dolphin_deflections_end.jpg`, with the respective histograms `hist_deflections_Dolphin.pdf`, `hist_deflections_end_Dolphin.pdf`), the wireframe representing the initial and final beam strain energy (files `Dolphin_energy_start.jpg`, `Dolphin_energy_end.jpg`, with the respective histograms `hist_energy_Dolphin.pdf`, `hist_energy_end_Dolphin.pdf`), the frequency histogram of the initial and final beam lengths (file `Dolphin_lengths.pdf`), the final inventory assignment histogram (file `Dolphin_stock_uniform_matches.pdf`), and the csv data for waste and new material descent curves (files `wastages.csv` and `new_material.csv`).

## Multi-stock optimization: Figure 14 
This task solves the optimization of an input gridshell shape, provided as a `.ply` file with fixed mesh topology, both in terms of static compliance reduction and reuse waste with respect to three different inventories of beam elements (`uniform`, `nonuniform1`, and `nonuniform2`).

More specifically, the command
~~~
python multistock_task_exec.py --device "cuda" --meshpath "meshes/Blob.ply" -niter 10000 --lr 5e-6 --saveinterval 100 --times --curves --hists --structhist --rendermodels --render --renderw --jpg
~~~
performs multi-stock computation of the shape `Blob.ply` and reproduces the images of Fig. 14. For brevity, let's consider only the case of the stock `nonuniform1`: in the `output\Blob_stock_nonuniform1` subfolder, the user can find the input and output gridshell (files `Blob_start.jpg` and `Blob_end.jpg`, respectively), the color-coded wireframes showing the initial and final beam assignment (files `Blob_batch_colors_start.jpg` and `Blob_batch_colors_end.jpg`), and the inventory assignment histograms for both input and output (files `Dolphin_stock_uniform_matches.pdf` and `Dolphin_stock_uniform_matches.pdf`).

## Multi-meshing optimization: Figure 15 and 16

<div style="border:1px solid #ff0000; padding:10px; border-radius:5px; background-color:rgb(247, 134, 134); color:rgb(0, 0, 0);">
<strong>Note:</strong> multi-meshing computation, even with CUDA, may require time! 
</div> 

This task at first computes different remeshings of a given target shape (provided as a `.ply`) with different average edge lengths, and then computes the optimization for each mesh both in terms of static compliance reduction and reuse waste from a given stock of beam elements.

More specifically, the command
~~~
python multimesh_task_exec.py --device "cuda" --meshpath "meshes/Blob.ply" -niter 10000 --lr 5e-6 --saveinterval 100 --times --stock "uniform" --curves --hists --structhist --rendermodels --render --renderw --jpg
~~~
performs multi-meshing computation of the shape `Blob.ply`, with respect the stock `uniform`, and reproduces the images of Fig. 16. For brevity, let's consider only the case of the average length 1.90: in the `output\Blob_1.90_stock_uniform` subfolder, the user can find the output gridshell (file `Blob_1.90_end.jpg`), the histogram of the initial and final beam assignment (file `Blob_1.90_start_end_matches.pdf`), evolutionary curves for matching_percentages, waste and new material (files `matching_percentages.pdf`, `wastages.pdf`, and `new_material.pdf`), the mesh in false colors representing the initial and final Service Load displacements (files `Blob_1.90_deflections_start.jpg`, `Blob_1.90_deflections_end.jpg`, with the respective histograms `hist_deflections_Blob_1.90.pdf`, `hist_deflections_end_Blob_1.90.pdf`), the wireframe representing the initial and final beam strain energy (files `Blob_1.90_energy_start.jpg`, `Blob_1.90_energy_end.jpg`, with the respective histograms `hist_energy_Blob_1.90.pdf`, `hist_energy_end_Blob_1.90.pdf`). The file `scatter_alt.pdf` of the parent directory contains the scatter plot of Fig. 15.

# Determinism of GPU-based computiations in PyTorch
Deterministic behavior of PyTorch CUDA is not guaranteed: output of linear algebra operation may slightly differ from a machine to another ([source](https://pytorch.org/docs/stable/notes/randomness.html)). However, by flagging `--reproducible` we can offer a full replicable implementation of our algorithm at the cost of slower computation time.