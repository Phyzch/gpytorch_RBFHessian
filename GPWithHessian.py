import numpy as np 
import torch 

def transform_training_target_into_1d_array(pots, grads, hessians, ndofs, fixdofs, M_H):
    '''
    transform the potential data (V^(1), ... , V^(M)) , gradient data: (dV^(1)/dx, ..., dV^(M)/dx) and Hessian data (d^2 V^(1)/dx^2, ..., d^2 V^(M_H)/dx^2) into a 1d array.
    This 1d array will obey multivariate normal distribution.
    The 1d array has the shape: (V^(1), ..., V^(M), dV^(1)/dx, dV^(2)/dx, ..., dV^(M)/dx, d^2 V^( h_1 )/dx^2, ..., d^2 V^( h_{M_H} )/dx^2).
    Note here the d^2 V/ dx^2 is upper triangular element of Hessian. the size is fH * (fH + 1) / 2, here fH = (natoms - len(fixatoms)) * 3.
    The data points that have hessian information is (h_1, ..., h_{M_H}).
    :param: pots: potential data: 1d numpy array. shape: M. Here M is number of data points.
    :param: grads: gradient data: 2d numpy array. shape: (M, d). Here M is number of data points, d is number of degrees of freedom.  d = 3 * natoms
    :param: hessians: Hessian data: shape ((natoms - len(fixatoms)) * 3, M_H * ( natoms - len(fixatoms) ) * 3).  here M_H is number of ab initio data point with Hessian information.
    :param: ndofs: number of degrees of freedom.
    :param: fixdofs: index of dofs to be fixed. The fixdofs will not be included in Hessians.
    :param: M_H: number of ab initio data point that has Hessian information. 
    '''
    if not isinstance(pots, np.ndarray): 
        raise RuntimeError("The pots are not numpy array when transform training data into 1d array.")

    if not isinstance(grads, np.ndarray):
        raise RuntimeError("The gradients are not numpy array when transform training data into 1d array.")
    
    if not isinstance(hessians, np.ndarray):
        raise RuntimeError("The hessians are not numpy array when transform training data into 1d array.")
    
    M = len(pots)  # number of data points with potential and gradient information

    grads_shape = np.shape(grads)
    # check shape of gradients
    if grads_shape[0] != M or grads_shape[1] != ndofs:
        raise RuntimeError("The shape of gradients do not match the number data point and natoms. number of data point {}, ndofs: {}. shape of grads: ({}, {})"
                           .format(M, ndofs, grads_shape[0], grads_shape[1]))

    # check shape of hessians.
    hessians_shape = np.shape(hessians)
    if hessians_shape[0] != ndofs - len(fixdofs) or hessians_shape[1] != M_H * (ndofs - len(fixdofs)):
        raise RuntimeError("The shape of hessians do not match the number of data points and natoms & fixatoms. number of data point with Hessian {}, ndofs: {}, number of fixdofs: {}, shape of hessians: ({}, {})"
                           .format(M_H, ndofs, len(fixdofs), hessians_shape[0], hessians_shape[1]))

    # transform pots, grads and hessian data into 1d array.
    y = pots 
    grads_1d = grads.flatten() 
    y = np.concatenate([y, grads_1d])

    iu = np.triu_indices( ndofs - len(fixdofs) )  # upper triangle indices
    hessians_new = np.reshape(hessians, (hessians_shape[0], M_H, hessians_shape[0]))
    hessians_new = np.transpose(hessians_new, (1, 0, 2))  # now the last two dimension is the dimension for hessian
    triu_hessian = np.array([ h[iu] for h in hessians_new ])
    triu_hessian_1d = triu_hessian.flatten() # 1d array (d^2 V(1)/ dx^2, ..., d^2 V(M_H)/ dx^2). the upper triangle part
    
    y = np.concatenate([y, triu_hessian_1d])

    return y 



