# Gaussian Process Regression for Hessian Prediction

This repository implements Gaussian Process Regression (GPR) for predicting Hessians (second derivatives) of scalar functions. The implementation is built on top of the GPyTorch framework.

## Background

Standard GPyTorch functionality supports:

- Function value prediction

- First-derivative (gradient) prediction

However, it does not natively support second-derivative (Hessian) prediction.

In our work:

https://pubs.acs.org/doi/10.1021/acs.jctc.5c00673

we require Hessian predictions for ring polymer instanton calculations, which are used to compute quantum tunneling effects in molecular systems. To enable this capability, we extended the GPyTorch framework to support analytical second derivatives of kernel functions.

This repository contains that extension.

## Key Features

✅ Gaussian Process regression for:

- Function values

- Gradients

- Hessians (new capability)

✅ Custom kernel implementation for second derivatives

✅ Compatible with both CPU and GPU (CUDA-enabled) execution

✅ Fully integrated with the GPyTorch training and inference pipeline

## Core Implementation

The central component of this repository is:

- ./library/RBFHessianKernel.py

This module implements a modified RBF (Radial Basis Function) kernel that analytically computes:

- Covariance between function values

- Covariance between gradients

- Covariance between Hessians

- Cross-covariances between all derivative orders

This extension enables consistent Gaussian Process training and prediction involving second-order derivatives.

## Hardware Support

The code automatically detects and uses:

- CPU by default

- GPU if CUDA is available

No modification is required from the user other than ensuring a CUDA-enabled PyTorch installation.

## Use Case

This implementation was specifically developed for:

- Ring polymer instanton calculations

- Quantum tunneling simulations

- Molecular potential energy surface modeling

- High-order derivative-informed Gaussian Process regression

It may also be useful for:

- Scientific machine learning applications requiring curvature information

- Optimization problems requiring second-order structure
