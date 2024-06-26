# Gamma-ray data analysis with Jax Prototype

This repo contains a prototype of a gamma-ray data analysis framework using Jax. The main goal is to asses the feasibility of using Jax as a backend to the (Gammapy)[https://gammapy.org/] library. It contains a set of benchmarks and tests to compare the performance of Jax with respect to the standard Gammapy analysis, which is based on Numpy and Scipy.

We would like to understand how performance scales with:

- The size of the dataset
- The complexity of the model (number of sources)
- Use of hardware accelerators (GPU, TPU)
- Use of gradients of the likelihood function
- Homogeneous vs heterogeneous datasets


## Setup 

To run the benchmarks in this repository you need to install the following dependencies:

```bash
mamba env create -f environment.yml
conda activate jax-gamma-ray-prototype
```

```bash
python prepare-test-datasets.py
```

```bash
python run-benchmarks.py
```
