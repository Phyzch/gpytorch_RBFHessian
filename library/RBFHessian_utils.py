import numpy as np 
import torch 

def transform_1d_train_targets_into_pots_grads_hessians(train_targets: torch.Tensor, M: int , ndofs: int, fixdofs: np.ndarray, M_H: int):
    '''
    Transform the 1d training targets from the pot, gradient and hessian data
    :param: M: total number of input data points.
    :param: ndofs: number of degrees of freedom in input data points.
    :param: fixdofs: numpy.ndarray. the dof that keep fixed in hessian calculation.
    :param: M_H: number of data points that contain hessian information.
    '''
    nactive = ndofs - len(fixdofs)
    hessian_triu_size = int(nactive * (nactive + 1) / 2) 
    targets_size = M * (ndofs + 1) + hessian_triu_size * M_H 
    
    batch_shape = train_targets.shape[:-2]
    assert train_targets.shape[-1] == targets_size, "the size of training target does not match the required data. Current shape: {}, required shape: {}".foramt(train_targets.shape[-1], targets_size)

    pot_data = train_targets[..., :M] 
    gradient_data = train_targets[..., M: M * (ndofs + 1)]
    hessian_data = train_targets[..., M * (ndofs + 1): M * (ndofs + 1) + hessian_triu_size * M_H]

    pots = pot_data 
    gradients = gradient_data.reshape([*batch_shape, M, ndofs])
    hessian_triu = hessian_data.reshape([*batch_shape, M_H, hessian_triu_size ])
    
    triu_indices = torch.triu_indices(nactive, nactive).to(device= train_targets.device)
    triu_1d_indices = triu_indices[0] * nactive + triu_indices[1]

    upper_triangular_hessians = torch.zeros([*batch_shape, M_H, nactive * nactive], device= train_targets.device)
    upper_triangular_hessians[..., triu_1d_indices] = hessian_triu 
    upper_triangular_hessians = torch.reshape(upper_triangular_hessians, [*batch_shape, M_H, nactive, nactive])

    hessians = upper_triangular_hessians.clone() 
    for i in range(M_H):
        upper_triangular_hessian_slice = upper_triangular_hessians[i]
        hessians[i] = upper_triangular_hessian_slice + upper_triangular_hessian_slice.t() - torch.diag(upper_triangular_hessian_slice.diag())
    
    
    return pots, gradients, hessians 
    
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

    triu_indices = torch.triu_indices(size1, size2).to(device= tensor.device)
    triu_1d_indices = (triu_indices[0] * size2 + triu_indices[1])
    # take upper triangular part of last two dimensions of Hessian.
    upper_triangle_tensor = torch.index_select(torch.reshape(tensor, (*batch_shape, size1 * size2) ), -1, triu_1d_indices)

    return upper_triangle_tensor