# filtered-point-process

Beta package associated with the paper "Filtered Point Processes Tractably Capture Rhythmic And Broadband Power Spectral Structure in Field-based Neural Recordings" (2024)

Patrick F. Bloniasz, Shohei Oyama, Emily P Stephen

## Package setup.  

### 1. Install required packages

While the pacakge is in beta we recommend engaging with the package in the following way:

#### Install Mamba through Conda

Mamba is a drop in replacement for conda, but is faster and is better at resolving dependency conflicts. 

```
conda install mamba -n base -c conda-forge
```

#### Create an isolated environment and install filtered-point-process

```
git clone https://github.com/Stephen-Lab-BU/filtered-point-process.git
cd filtered-point-process
mamba env create -f environment.yml
mamba activate filtered-point-process
python -m pip install git+https://github.com/Stephen-Lab-BU/filtered-point-process.git

```

#### Major upcoming changes

The package is being refactored to have a set of tutorials and be built on a multivariate version of this package. Currently, the package is set up to do univariate models. The full multivariate build with tutorials will be released as a complete build on November 27th. 