## Predicting Hessian using Gaussian Process Regression.
This code predicts Hessians of function using Gaussian Process Regression. The code is built on GPytorch (https://gpytorch.ai/). 
The conventional GPytorch code is only capable of predicting gradients (derivative of potentials). In our work (https://chemrxiv.org/engage/chemrxiv/article-details/67f9c3fc81d2151a021107e1), we try to predict Hessians for the ring polymer instanton calculation. To achieve this goal, I developed this code to predict Hessians of potentials using Gaussian Process Regression based on the code in GPytorch.

The key module is ./library/RBFHessianKernel.py which I rewrite the kernel function for predicting Hessians.



