# Radial Basis function kernel with Hessian information.
# Adapted from rbf_kernel_grad.py in gpytorch
# This is an extension of gpytorch rbfkernel
import numpy as np 
import torch 
from linear_operator.operators import KroneckerProductLinearOperator

from gpytorch.kernels.rbf_kernel import postprocess_rbf, RBFKernel 


def take_upper_triangular_part(tensor):
    '''
    Take the upper triangular part of the matrix of tensor (upper triangular part of last 2 dimensions).
    :param: tensor: tensor to take the upper triangular part.
    '''
    batch_shape = tensor.shape[:-2]
    size1 = tensor.shape[-2]
    size2 = tensor.shape[-1]

    if size1 != size2:
        raise RuntimeError("The matrix we take upper triangular part about is not square matrix.  size1: {},  size2: {}".format(size1, size2))

    hessian_triu_size = size1 * (size1 + 1) / 2
    # take upper triangular part of last two dimensions of Hessian.
    upper_triangule_tensor = torch.triu(tensor)
    # Remove zero part
    upper_triangule_tensor = upper_triangule_tensor[ upper_triangule_tensor != 0]
    # reshape the tensor to the desired shape.
    upper_triangule_tensor = upper_triangule_tensor.reshape(*batch_shape, hessian_triu_size)

    return upper_triangule_tensor

def matrix_outer_product_combination(matrix1, matrix2, d):
    '''
    two matrix with same shape (d,d). We compute the tensor which is outer product of two matrices.
    The index for dimension will combine in different ways: Let the dimension of matrix 1 be (d1,d1), the dimension of matrix 2 be (d2, d2), 
    then the tensor will be the sum of the outer product of two matrices with the combinationsof all dimensions.
    (d1,d1,d2,d2), (d1,d2,d1,d2), (d1,d2,d2,d1), (d2,d1,d1,d2), (d2,d1,d2,d1), (d2,d2, d1,d1)
    '''
    matrix1_size1 = matrix1.shape[-2]
    matrix1_size2 = matrix1.shape[-1]
    matrix2_size1 = matrix2.shape[-2]
    matrix2_size2 = matrix2.shape[-1]
    assert matrix1_size1 == d and matrix1_size2 == d and matrix2_size1 == d and matrix2_size2 == d, "Size of the matrix not equal to the input dimension d"

    # shape: (d,d, 1, 1)
    matrix1_extend = matrix1.unsqueeze(-1).unsqueeze(-1)
    # shape: (1,1,d,d)
    matrix2_extend = matrix2.unsqueeze(-3).unsqueeze(-3)
    # shape: (d, d, d, d).  order: (d1,d1,d2,d2)
    outer_product1 = matrix1_extend * matrix2_extend 
    # order: (d1, d2, d1, d2)
    outer_product2 = outer_product1.transpose(-2, -3)
    # order: (d1,d2,d2,d1)
    outer_product3 = outer_product2.transpose(-1, -2)
    # order: (d2, d1, d1, d2)
    outer_product4 = outer_product2.transpose(-3, -4)
    # order: (d2, d1, d2, d1)
    outer_product5 = outer_product3.transpose(-3, -4)
    # order: (d2, d2, d1, d1)
    outer_product6 = outer_product5.transpose(-2,-3)

    outer_product = outer_product1 + outer_product2 + outer_product3 + outer_product4 + outer_product5 + outer_product6 

    return outer_product 

def matrix_outer_product_same_operator_combination(matrix1, matrix2, d):
    '''
    two matrix with same shape (d,d). We compute the tensor which is outer product of two matrices.
    The index for dimension will combine in different ways: Let the dimension of matrix 1 be (d1,d1), the dimension of matrix 2 be (d2, d2).
    then the tensor will be the sum of the outer product of two matrices with the combinationsof all dimensions.
    (d1,d1,d2,d2), (d1,d2,d1,d2), (d1,d2,d2,d1).  (We treat (d1,d1, d2, d2) the same as (d2, d2, d1, d1), because the matrix 1 and matrix 2 here represent the same operator).
    '''
    matrix1_size1 = matrix1.shape[-2]
    matrix1_size2 = matrix1.shape[-1]
    matrix2_size1 = matrix2.shape[-2]
    matrix2_size2 = matrix2.shape[-1]
    assert matrix1_size1 == d and matrix1_size2 == d and matrix2_size1 == d and matrix2_size2 == d, "Size of the matrix not equal to the input dimension d"

    # shape: (d,d, 1, 1)
    matrix1_extend = matrix1.unsqueeze(-1).unsqueeze(-1)
    # shape: (1,1,d,d)
    matrix2_extend = matrix2.unsqueeze(-3).unsqueeze(-3)
    # shape: (d, d, d, d).  order: (d1,d1,d2,d2)
    outer_product1 = matrix1_extend * matrix2_extend 
    # order: (d1, d2, d1, d2)
    outer_product2 = outer_product1.transpose(-2, -3)
    # order: (d1,d2,d2,d1)
    outer_product3 = outer_product2.transpose(-1, -2)

    outer_product = outer_product1 + outer_product2 + outer_product3
    return outer_product 

def matrix_outer_product(matrix1, matrix2, d):
    '''
    compute the outer product of two matrices.
    '''
    matrix1_size1 = matrix1.shape[-2]
    matrix1_size2 = matrix1.shape[-1]
    matrix2_size1 = matrix2.shape[-2]
    matrix2_size2 = matrix2.shape[-1]
    assert matrix1_size1 == d and matrix1_size2 == d and matrix2_size1 == d and matrix2_size2 == d, "Size of the matrix not equal to the input dimension d"

    # shape: (d,d, 1, 1)
    matrix1_extend = matrix1.unsqueeze(-1).unsqueeze(-1)
    # shape: (1,1,d,d)
    matrix2_extend = matrix2.unsqueeze(-3).unsqueeze(-3)

    outer_product = matrix1_extend * matrix2_extend 

    return outer_product 


class RBFKernelHessian(RBFKernel):
    r"""

    :param: 
    """

    def forward(self, x1, x2, M_H1= None, M_H2= None, hessian_data_point_index_1= None,  hessian_data_point_index_2= None, ndofs= None, fixdofs= None):
        '''
        compute the covariance matrix between two 1d array data:
        y1 = (V^(1), ..., V^(M), dV^(1)/dx, dV^(2)/dx, ..., dV^(M)/dx, d^2 V^(1)/dx^2, ..., d^2 V^(M_H)/dx^2).
        y2 : similarly but with different data number M and Hessian data number M_H. 
        Here d^2 V^(1) / dx^2 is the upper triangular part of Hessian with active dimensions. We do not include all dimensions in the Hessian (exclude fixdof). The size of hessian matrix is native * (nactive + 1) / 2
        :param: x1: input data, shape: (M1, d).
        :param: x2: input data, shape: (M2, d)
        :param: M_H1: data number that contains Hessian information in x1.
        :param: M_H2: data number that contains Hessian information in x2. 
        :param: hessian_data_point_index_1: the index of data points that contain hessian information. points (h_1, .., h_{M_H1}) have hessian info. numpy array.
        :param: hessian_data_point_index_2: the index of data points that contain hessian information. points (h_1, ...,h_{M_H2}) have hessian info. numpy array.
         We assume first M_H data points contain Hessian, thus the data points with Hessian information is put in front
        :param: ndofs: number of degrees of freedom.
        :param: fixdofs: index of dofs to be fixed in hessian. The fix dofs will not be included in Hessians.
        
        '''
        if M_H1 == None:
            raise RuntimeError("Must provide M_H1 (number of data points with Hessian) when using RBFKernelHessian.")
        
        if M_H2 == None:
            raise RuntimeError("Must provide M_H2 (number of data points with Hessian) when using RBFKernelHessian.")
        
        if hessian_data_point_index_1 == None:
            raise RuntimeError("Must provide hessian_data_point_index1 (index of hessian data points that contain hessian information) when using RBFKernelHessian")
        elif type(hessian_data_point_index_1) != np.ndarray and type(hessian_data_point_index_1) != torch.Tensor:
            raise RuntimeError("the data type of hessian_data_point_index_1 must be either numpy.ndarray or torch.Tensor. The data type now is: " + str(type(hessian_data_point_index_1)))

        if hessian_data_point_index_2 == None:
            raise RuntimeError("Must provide hessian_data_point_index2 (index of hessian data points that contain hessian information) when using RBFKernelHessian")
        elif type(hessian_data_point_index_2) != np.ndarray and type(hessian_data_point_index_2) != torch.Tensor:
            raise RuntimeError("the data type of hessian_data_point_index_2 must be either numpy.ndarray or torch.Tensor. The data type now is: " + str(type(hessian_data_point_index_2)))

        if ndofs == None:
            raise RuntimeError("Must provide ndofs (total number of degrees of freedom) when using RBFKernelHessian.")
         
        if fixdofs == None:
            raise RuntimeError("Must provide fixdofs (index of dofs to be fixed) when using RBFKernelHessian")
        
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        M1, d = x1.shape[-2:]  # M1: number of data points in x1
        M2 = x2.shape[-2]  # M2: number of data points in x2

        assert d == ndofs, "ndofs does not equal to size of x1"

        if M_H1 > M1 or M_H2 > M2:
            raise RuntimeError("The number of data points containing Hessian can't be larger than total number of data points. M1: {}, M2: {}, M_H1: {}, M_H2:{}".format(M1, M2, M_H1, M_H2))
        
        nactive = ndofs - len(fixdofs)  # number of active dofs
        hessian_triu_size = nactive * (nactive + 1) / 2 

        active_dims = np.delete(np.arange(ndofs), fixdofs) # index of active dims

        x1WithHessian = x1[..., hessian_data_point_index_1, :] # x1 data points that contain Hessian information. We assume M_H1 is the same across batches.
        x2WithHessian = x2[..., hessian_data_point_index_2, :] # x2 data points that contain Hessian information. We assume M_H2 is the same across batches.

        y1_len = M1 * (ndofs + 1) + M_H1 * (nactive * (nactive + 1))/2  # pot + grad + up-triangular of hessian.
        y2_len = M2 * (ndofs + 1) + M_H2 * (nactive * (nactive + 1))/2 
        
        K = torch.zeros(*batch_shape, y1_len, y2_len, device = x1.device, dtype = x1.dtype)

        # Code below compute the covariance between 1d data containing pots, gradient and hessian.
        # Scale the inputs by the lengthscale (for stability)
        x1_ = x1.div(self.lengthscale)   # self.lengthscale shape: [1, d], x1 shape: [M, d]
        x2_ = x2.div(self.lengthscale) 

        # outer: (x1 - x2)/(l_k)^2.  shape: (batch_shape, M1, M2, d)
        outer = x1_.reshape(*batch_shape, M1, 1, d) - x2_.reshape(*batch_shape, 1, M2, d) 
        outer = outer / self.lengthscale.unsqueeze(-2)  
        
        # 1) Kernel block. Squared exponential kernel.  K_11 shape: (M1, M2)
        diff = self.covar_dist(x1_, x2_, square_dist= True)
        K_11 = postprocess_rbf(diff)  # .div_(-2).exp_(). exp(-1/2 * diff)
        K[..., :M1, :M2] = K_11 

        # 2) First Gradient block: K12: Covariance between (V^(1), .., V^(M1)) and (dV^(1)/dx, ..., dV^(M2)/dx) (go with dof index first)
        # shape: (M1, M2 * d).  reshape: (M1, M2, d) -> (M1, M2 * d)
        outer1 = outer.reshape(*batch_shape, M1, M2 * d)
        # (M1, M2) -> (M1,M2,1) -> (M1,M2,d) -> (M1,M2 *d)
        K_11_extend1 = K_11.unsqueeze(-1).repeat( [ *([1] * n_batch_dims), 1, 1, d] ).reshape( *batch_shape, M1 , M2 * d)  # along last dimension: same squared exponential function repeat d times, then squared exponential for the next element. 
        K_12 = K_11_extend1 * outer1 
        K[..., :M1, M2: M2 * (1 + d)] = K_12 

        # 3) Second Gradient block: K21: Covariance between (dV^(1)/dx, .., dV^(M1)/dx) (go with dof index first)  and (V^(1), ..., V^(M2))  
        # # shape: (M1 * d, M2). reshape: (M1, d, M2) -> (M1 * d, M2)
        # outer2 = torch.transpose(outer, (-1, -2))
        # outer2 = outer2.reshape(*batch_shape, M1 * d, M2)
        # # (M1,M2) -> (M1, 1, M2) -> (M1, d, M2) -> (M1 * d, M2)
        # K_11_extend2 = K_11.unsqueeze(-2).repeat( [ *([1] * n_batch_dims), 1, d, 1 ] ).reshape(*batch_shape, M1 * d, M2)
        # K_21 = - outer2 * K_11_extend2 
        
        K_21 = - torch.transpose(K_12.reshape(*batch_shape, M1, M2, d), (-1, -2)).reshape(*batch_shape, M1 * d, M2) 
        K[..., M1 : M1 * (1 + d), : M2] = K_21 

        # 4) K22: Hessian block.  covariance function between (dV^(1)/dx, ..., dV^(M1)/dx), and (dV^(1)/dx, ..., dV^(M2)/dx).   shape: (M1 * d, M2 * d)\
        # outer3 = (x1_n - x2_n)/ (l_n)^2 * (x1_k - x2_k)/(l_k) ^2
        # shape: (M1, M2, 1, d) * (M1, M2, d, 1) -> (M1, M2, d, d) -> (M1, d, M2, d) -> (M1 *d, M2 * d)
        outer3 = torch.transpose(outer.unsqueeze(-2) * outer.unsqueeze(-1), (-3, -2) ).reshape(*batch_shape, M1* d, M2* d)
        # (M1,M2) -> (M1, 1, M2, 1) -> (M1, d, M2, d) -> (M1 * d, M2 * d)
        K_11_extend3 = K_11.unsqueeze(-1).unsqueeze(-3).repeat([ *([1] * n_batch_dims), 1, d, 1, d ]).reshape(*batch_shape, M1* d, M2* d)
        # shape : (M1 * d, M2 * d)
        kp = KroneckerProductLinearOperator(
            torch.ones(M1, M2, device= x1.device, dtype= x1.dtype).repeat(*batch_shape, 1, 1),
            torch.eye(d,d, device= x1.device, dtype= x1.dtype).repeat(*batch_shape, 1, 1) / torch.pow(self.lengthscale,2)
        )

        chain_rule = kp.to_dense() - outer3 
        K_22 = chain_rule * K_11_extend3 
        K[..., M1: M1 * (1+d), M2: M2 * (1+d)] = K_22

        # variable related to the hessian.
        x1WithHessian_ = x1WithHessian.div(self.lengthscale)
        x2WithHessian_ = x2WithHessian.div(self.lengthscale)

        RBF_kernel_1 = postprocess_rbf(self.covar_dist(x1_, x2WithHessian_, square_dist= True))  # squared exponential kernel between x1 and x2WithHessian
        RBF_kernel_2 = postprocess_rbf(self.covar_dist(x1WithHessian_, x2_, square_dist= True)) # squared exponential kernel between x1WithHessian and x2.

        # 5) K13: covariance function between (V^(1), ..., V^(M1)) and (d^2 V^(1)/dx^2, ..., d^2 V^(M_H2)/ dx^2)
        # K13 for full Hessian, it's the case when we include all dofs and have M_H = M (have Hessian for all data points). shape: (M1, M2, d, d)
        K13_full = - torch.transpose(K_22.reshape(*batch_shape, M1, d, M2, d), (-2, -3))
        # truncate the data number in x2 until M_H2, exclude fix dof from Hessian. 
        K13_partial = K13_full[..., :, hessian_data_point_index_2, active_dims, active_dims]
        K13_upper_triangle = take_upper_triangular_part(K13_partial)
        # reshape it to the desired shape: (M1 , M_H2 * hessian_triu_size)
        K13 = K13_upper_triangle.reshape(*batch_shape, M1, M_H2 * hessian_triu_size)

        K[..., :M1, M2 * (d+1):] = K13 

        # 6) K31 : covariance function between ( d^2 V^(1) / dx^2, ...,  d^2 V^(M_H1)/dx^2 ) and (V^(1), ..., V^(M2))
        K31_full = K13_full  # shape: (M1, M2, d, d)
        # truncate the data number in x1 until M_H1, exclude fix dof from Hessian
        K31_partial = K31_full[..., hessian_data_point_index_1, :, active_dims, active_dims]
        K31_upper_triangle = take_upper_triangular_part(K31_partial)
        K31 = torch.transpose(K31_upper_triangle, (-1, -2)).reshape(*batch_shape, M_H1 * hessian_triu_size, M2)

        K[..., M1 * (1+d): , :M2] = K31 

        # 7): K23: covariance function between (dV^(1)/dx, ..., dV^(M1)/dx) and (d^2 V^(1)/dx^2, ..., d^2 V^(M_H2)/ dx^2)
        # K13_extend shape: (M1, d, M_H2 , hessian_triu_size)
        K13_extend = K13.unsqeueze(-2).repeat([  *([1] * n_batch_dims), 1, d, 1]).reshape(*batch_shape, M1, d, M_H2, hessian_triu_size)
        # (x1 - x2)/(l_k)^2.  shape: (batch_shape, M1, M_H2, d)
        outer_part23 = outer[..., :, hessian_data_point_index_2, :] 
        # outer23: (M1, M_H2, d) -> (M1, d, M_H2) -> (M1, d, M_H2, 1)
        outer23 = torch.transpose(outer_part23, (-1, -2)).unsqueeze(-1)
        # - (x1_k - x2_k)/(l_k)^2 * K_13
        K23_part1 = -outer23 * K13_extend
        K23_part1 = K23_part1.reshape(*batch_shape, M1 * d, M_H2 * hessian_triu_size)

        # second part of K23: (delta_kn /(l_n)^2 * (x1_p - x2_p)/(l_p)^2 + delta_kp / (l_p)^2 * (x1_n - x2_n)/(l_n)^2) * exp(- \sum_m (x_m^(i) - x_m^(j))^2 / (2 * (l_m)^2))
        # 1/l^2 shape: (M1, M_H2, d, d, 1)
        kronecker1 = (torch.eye(d,d, device= x1.device, dtype= x1.dtype) / torch.pow(self.lengthscale, 2)).repeat(*batch_shape, M1, M_H2, 1, 1).unsqueeze(-1)
        # shape: (M1,M_H2, 1, 1, d).  (x1_p - x2_p) / (l_p)^2
        outer_part23_1 = outer_part23.unsqueeze(-2).unsqueeze(-2)
        kp_part1 = kronecker1 * outer_part23_1

        #1/l^2  shape: (M1, M_H2, d, 1, d)
        kronecker2 = torch.transpose(kronecker1, (-1, -2))
        # shape: (M1, M_H2, 1, d, 1)
        outer_part23_2 = torch.transpose(outer_part23_1, (-1, -2)) 
        kp_part2 = kronecker2 * outer_part23_2
        # delta_kn /(l_n)^2 * (x1_p - x2_p)/(l_p)^2 + delta_kp / (l_p)^2 * (x1_n - x2_n)/(l_n)^2. shape: (M1, M_H2, d, d, d)
        kp_part = kp_part1 + kp_part2  

        # take active_dim part of hessian from kp_part and then take upper triangular part. shape: (M1, M_H2, d, nactive, nactive)
        kp_part = kp_part[..., active_dims, active_dims]
        kp_part_upper_triangule = take_upper_triangular_part(kp_part)
        # shape: (M1* d, M_H2* hessian_triu_size)
        kp_part_upper_triangule = torch.transpose(kp_part_upper_triangule, (-2, -3)).reshape(*batch_shape, M1 * d, M_H2 * hessian_triu_size)

        # RBF_K23 = squared exponential function between two inpus in x1 and x2
        # shape: (M1, MH_2) ->  (M1, 1, MH_2, 1) -> (M1, d, MH_2, hessian_size) -> (M1 *d, MH_2 * hessian_size)
        RBF_K23 = RBF_kernel_1.unsqueeze(-1).unsqueeze(-3).repeat([ *([1] * n_batch_dims), 1, d, 1, hessian_triu_size ]).reshape(*batch_shape, M1 * d, M_H2 * hessian_triu_size)

        K23_part2 = RBF_K23 * kp_part_upper_triangule

        K23 = K23_part1 + K23_part2 

        K[..., M1 : M1 * (1 + d), M2 * (1+d) : ] = K23 

        # 8): K32: covariance function between (d^2 V^(1) / dx^2, .., d^2 V^(M_H2) / dx^2) and (dV^(1)/dx, .., dV^(M1)/dx)        
        # first part of K32: (x1_p - x2_p)/(l_p)^2 * K31
        # K31 extend shape: (M_H1, hessian_triu_size, M2, d)
        K31_extend = K31.unsqueeze(-1).repeat( [ *([1] * n_batch_dims), 1, 1, d] ).reshape(*batch_shape, M_H1, hessian_triu_size, M2, d)
        # (x1 - x2)/ (l_p)^2. shape: (MH1, M2, d)
        outer_part32 = outer[..., hessian_data_point_index_1, :, :]
        # outer32: (MH1, M2, d) -> (MH1, 1, M2, d)
        outer_32 = outer_part32.unsqueeze(-3) 
        # (x1 - x2)/ (l_p)^2 * K31 
        K32_part1 = outer_32 * K31_extend 
        K32_part1 = K32_part1.reshape(*batch_shape, M_H1 * hessian_triu_size, M2 * d)

        # second part of K32: - (\delta_kp * (x1_n - x2_n) + \delta_np (x1_k - x2_k)) / ((l_k)^2 * (l_n)^2 ) * RBF_kernel. here (k,n) is index of hessian , p is index for gradient. 
        # 1/l^2. shape: (d,d) -> (M_H1, M2, d, d) -> (M_H1, M2, d, 1, d)
        kronecker1 = (torch.eye(d,d, device= x1.device, dtype= x1.dtype) / torch.pow(self.lengthscale, 2)).repeat(*batch_shape, M_H1, M2, 1, 1).unsqueeze(-2)
        # shape (M_H1, M2, d) -> (M_H1, M2, d, 1) -> (M_H1, M2, 1, d, 1)
        outer_part32_1 = outer_part32.unsqueeze(-1).unsqueeze(-3) 
        kp_part1 = kronecker1 * outer_part32_1 

        # 1/l^2. shape(M_H1,  M2, 1, d, d)
        kronecker2 = torch.transpose(kronecker1, (-2, -3))
        # shape(M_H1, M2, d, 1, 1)
        outer_part32_2 = torch.transpose(outer_part32_1, (-2, -3))
        kp_part2 = kronecker2 * outer_part32_2 
        # - (\delta_kp * (x1_n - x2_n) + \delta_np (x1_k - x2_k)) / ((l_k)^2 * (l_n)^2 ). shape: (M_H1, M2, d, d, d). the dimension (-2,-3) is the index for hessian.
        kp_part = - kp_part1 - kp_part2 

        # transpose tensor to make the dof index of hessian the last two dimension
        kp_part = torch.transpose(torch.transpose(kp_part, (-1, -2)), (-2, -3))
        # take active dim part of hessian from kp_part and then take upper triangular part. shape (M_H1, M2, nactive, nactive, d)
        kp_part = kp_part[..., active_dims, active_dims]
        kp_part_upper_triangule = take_upper_triangular_part(kp_part)
        # shape: (M_H1 * hessian_triu_size, M2 * d)
        kp_part_upper_triangule = torch.transpose(torch.transpose(kp_part_upper_triangule, (-1, -2)), (-2, -3)).reshape(*batch_shape, M_H1 * hessian_triu_size, M2 * d)

        # RBF_K32: squared exponential function between x1 and x2
        RBF_K32 = RBF_kernel_2.unsqueeze(-1).unsqueeze(-3).repeat([  *([1] * n_batch_dims), 1, hessian_triu_size, 1, d ]).reshape(*batch_shape, M_H1 * hessian_triu_size, M2 * d)

        K32_part2 = RBF_K32 * kp_part_upper_triangule

        K32 = K32_part1 + K32_part2 
        K[..., (1 + d) * M1: , M2 : M2 * (d + 1) ] = K32 

        # 9) K33: covariance function between hessian matrices. 
        # (x1_p - x2_p)/(l_p)^2. Note only some data points have hessian information. Also only active_dims dofs will appear in hessian matrix.
        K33_outer_part = outer[..., hessian_data_point_index_1, hessian_data_point_index_2, active_dims]
        # (x1_p - x2_p)/(l_p)^2 * (x1_q - x2_q)/(l_q)^2
        K33_outer_part_product = K33_outer_part.unsqueeze(-1) * K33_outer_part.unsqueeze(-2)
        # \delta_pq / (l_p)^2
        kronecker = (torch.eye(nactive ,nactive , device= x1.device, dtype= x1.dtype) / torch.pow(self.lengthscale, 2)).repeat(*batch_shape, M_H1, M_H2, 1, 1)

        # (x1_p - x2_p)/(l_p)^2 * (x1_q - x2_q)/(l_q)^2 * (x1_k - x2_k)/(l_k)^2 * (x1_n - x2_n)/(l_n)^2
        K33_part1 = matrix_outer_product(K33_outer_part_product, K33_outer_part_product, nactive)

        # -(\delta_pq / (l_p)^2) * (x1_k - x2_k)/(l_k)^2 * (x1_n - x2_n)/(l_n)^2  + their other index combinations
        K33_part2 = - matrix_outer_product_combination(K33_outer_part_product, kronecker, nactive)

        # \delta_pq / (l_p)^2 * \delta_kn / (l_k)^2  + \delta_kp / (l_p)^2 * \delta_nq / (l_q)^2 + \delta_np / (l_p)^2 * \delta_kq / (l_q)^2
        K33_part3 = matrix_outer_product_same_operator_combination(kronecker, kronecker, nactive)
        # shape: (*batch_shape, M_H1, M_H2, nactive, nactive, nactive, nactive)
        K33 = K33_part1 + K33_part2 + K33_part3 

        # shape: (*batch_shape, M_H1, M_H2, nactive, nactive, hessian_size)
        K33 = take_upper_triangular_part(K33)
        # shape: (*batch_shape, M_H1, M_H2, hessian_size, nactive, nactive)
        K33 = torch.transpose(torch.transpose(K33, (-1, -2)), (-2, -3)) 
        # shape: (*batch_shape, M_H1, M_H2, hessian_size, hessian_size)
        K33 = take_upper_triangular_part(K33)

        K[..., (1 + d) * M_H1: , (1 + d) * M_H2 : ] = K33 

        return K 
        

    


