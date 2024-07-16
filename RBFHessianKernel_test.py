from RBFKernelHessian.RBFHessianKernel import RBFKernelHessian
import numpy as np 
import torch 
from gpytorch import settings 

# test the RBFHessianKernel with small data.

def initialize_training_inputs():
    '''
    '''
    d = 4  # dof of input data is set to be 3
    fixdofs = torch.tensor([1])   # dof 1 is set to be fixed when computing hessian.
    
    M1 = 4
    hessian_data_point_index_1 = torch.tensor([1])
    x1 = np.random.random([M1, d])

    M2 = 5
    hessian_data_point_index_2 = torch.tensor([2,3])
    x2 = np.random.random([M2, d])

    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)

    return x1, x2, hessian_data_point_index_1, hessian_data_point_index_2, d, fixdofs 

def test_RBFHessianKernel():
    '''
    test the class RBFHessianKernel we have defined.
    '''
    x1, x2, hessian_data_point_index_1, hessian_data_point_index_2, d, fixdofs = initialize_training_inputs()

    base_kernel = RBFKernelHessian(ard_num_dims= d, hessian_fixdofs= fixdofs)

    # we need to set settings.lazily_evaluate_kernels(False) to enable computing the function
    with settings.lazily_evaluate_kernels(False):
        covar_x = base_kernel(x1, x2, hessian_data_point_index_1= hessian_data_point_index_1, hessian_data_point_index_2= hessian_data_point_index_2)

    
test_RBFHessianKernel()


