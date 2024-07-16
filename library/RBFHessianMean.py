import torch
from gpytorch.means.mean import Mean 

class ConstantMeanHessian(Mean):
    '''
    module that represents the mean function for data with Hessian information
    the mean function is a constant value.
    '''
    def __init__(self, prior= None, batch_shape= torch.Size(), **kwargs):
        super(ConstantMeanHessian, self).__init__()
        self.batch_shape = batch_shape 
        self.register_parameter(name="constant", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")
    
    def forward(self, input, M_H= None, nactive= None):
        '''
        input shape: [N, d]
        mean function shape: [V1, .., V^(N), dV^(1)/dx, .., dV^(N)/dx, dV^(h_1)/dx^2, .., dV^(h_MH)/dx^2 ]
        :param: M_H: number of data points with Hessian information.
        :param: nactive: active dimensions for computing hessian.
        '''
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2]) 

        N = input.size(-2)
        d = input.size(-1)

        # size of data points for function, gradient and hessian information.
        func_size = N 
        grad_size = N * d
        hessian_size = M_H * nactive * (nactive + 1) / 2

        total_size = func_size + grad_size + hessian_size 

        mean = self.constant.expand(*batch_shape, total_size).contiguous() 
        mean[..., func_size: ] = 0 

        return mean 
    