import numpy as np 
import torch 
import gpytorch 
from library.RBFHessian_gp import GPModelWithHessians 
from library.RBFHessian_utils import transform_1d_train_targets_into_pots_grads_hessians
import library.RBFHessian_gp as RBFHessian_gp

def prepare_likelihood_noise(ndofs, hessian_triu_size):
    '''
    generate the estimated noise for potential, force and hessian.
    pot_noise: shape: [1]
    force_noise: shape: [ndofs]
    hessian_noise: shape: [hessian_triu_size]
    '''
    # the variance of the noise.
    pot_noise = np.power(np.array([np.power(10.0, -3)]), 2)
    force_noise = np.power(np.ones([ndofs]) * np.power(10.0, -2), 2)
    hessian_noise = np.power(np.ones([hessian_triu_size]) * np.power(10.0, -2), 2)


    return pot_noise, force_noise, hessian_noise 

def prepare_kernel_prior(ndof, gpr_SE_kernel_number):
    '''
    generate the prior distribution of the kernel function
    '''
    kernel_outputscale = np.ones([gpr_SE_kernel_number]) * 0.04 
    kernel_length_scale_ratio = np.ones([gpr_SE_kernel_number]) * 0.3

    return kernel_outputscale, kernel_length_scale_ratio

def franke(X: torch.Tensor, Y: torch.Tensor):
    '''
    potential and gradient of franke function. https://www.sfu.ca/~ssurjano/franke2d.html
    '''
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    gradient = torch.cat( (dfx.unsqueeze(-1), dfy.unsqueeze(-1)), dim= 1)
    gradient = gradient.reshape(gradient.numel())

    return f, gradient

def franke_hessian(X: torch.Tensor,Y: torch.Tensor):
    '''
    hessian of franke function. https://www.sfu.ca/~ssurjano/franke2d.html
    '''
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    dfxx = -81/2 * term1 + 81/4 * (9 * X - 2).pow(2) * term1 + \
            -18 * 9 / 49 * term2 + (18 / 49 * (9 * X + 1)).pow(2) * term2 + \
            - 81/2 * term3 + 81 / 4 * (9 * X - 7).pow(2) * term3 + \
            18 * 9 * term4  - (18 * (9 * X - 4)).pow(2) * term4 
    
    dfxy = (-9/2 * (9 * X - 2)) * (-9/2 * (9 * Y - 2)) * term1 + \
            (-18/49 * (9 * X + 1)) * (-9 / 10) * term2 + \
            (-9/2 * (9 * X - 7)) * (-9/2 * (9 * Y - 3)) * term3 + \
            + 18 * (9 * X - 4) * (-18 * (9 * Y - 7)) * term4 
    
    dfyy = (-9/2 * 9) * term1 + (9/2 * (9 * Y -2)).pow(2) * term1 + \
            pow(9/10, 2) * term2 + (-9 / 2 * 9) * term3 + (9 / 2 * (9 * Y - 3)).pow(2) * term3 + \
           18 * 9 * term4 - (18 * (9 * Y - 7)).pow(2) * term4 
    
    hessian_2d = torch.cat( (dfxx.unsqueeze(-1), dfxy.unsqueeze(-1), dfyy.unsqueeze(-1)), dim= 1)
    hessian = hessian_2d.reshape( hessian_2d.numel() )
    return hessian

def prepare_training_inputs():
    '''
    generate the input and targets of the training data.
    Use 2d franke function to use as input data. See: https://www.sfu.ca/~ssurjano/franke2d.html
    '''    
    ndof = 2 
    hessian_fixdofs = torch.tensor([])
    nactive = int(ndof - len(hessian_fixdofs))
    hessian_triu_size = int((nactive + 1) * nactive / 2)

    xv, yv = torch.meshgrid(torch.linspace(0,1,10) , torch.linspace(0,1,10), indexing = 'ij' )
    
    M = xv.numel()
    hessian_data_point_index = torch.arange(0, 100, 3)
    
    train_inputs = torch.cat(
        (
        xv.contiguous().view(xv.numel(), 1),
        yv.contiguous().view(yv.numel(), 1)
        ),
        dim = 1
    )
    train_x_with_hessian = torch.index_select(train_inputs, dim= 0, index= hessian_data_point_index)
    f, gradient = franke(train_inputs[:, 0], train_inputs[:, 1])
    hessian = franke_hessian(train_x_with_hessian[:, 0], train_x_with_hessian[:, 1])
    
    # add noise to the function, gradient and hessian.  The value here should match perpare_likelihood_noise()
    pot_noise = np.power(10.0, -3)
    force_noise = np.power(10.0, -2)
    hessian_noise = np.power(10.0, -2)

    f = f + pot_noise * torch.rand(len(f))
    gradient = gradient + force_noise * torch.rand(len(gradient))
    hessian = hessian + hessian_noise * torch.rand(len(hessian))

    target_len = int(M * (ndof + 1) + hessian_triu_size * len(hessian_data_point_index))
    train_targets  = torch.cat( (f, gradient, hessian) , dim= 0 )
    assert len(train_targets) == target_len, "the length of target data is wrong."

    return train_inputs, train_targets, hessian_fixdofs, hessian_data_point_index 

def prepare_test_data(test_data_number, test_data_hessian_data_point_index):
    '''
    prepare the test_inputs and test target for the 2d franke function.
    '''
    ndofs = 2 
    test_inputs = torch.rand(test_data_number, ndofs)
    test_inputs_with_hessian = torch.index_select(test_inputs, dim= 0, index= test_data_hessian_data_point_index)

    f, gradient = franke(test_inputs[:, 0], test_inputs[:, 1])
    hessian = franke_hessian(test_inputs_with_hessian[:, 0], test_inputs_with_hessian[:, 1])

    test_targets = torch.cat((f, gradient, hessian), dim= 0)

    return test_inputs, test_targets




def model_training(model:GPModelWithHessians, train_inputs: torch.Tensor, train_targets: torch.Tensor):
    training_iter = 1000 

    # Find optimal model hyper-parameter 
    likelihood = model.likelihood 
    model.train() 
    likelihood.train() 

    # Use the adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.05)

    # "Loss" for GPS -- the marginal log likelihood 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    M = train_inputs.shape[-2]  # total number of data points.
    M_H = len(model.training_data_hessian_data_point_index)  # number of data points have hessian info

    for i in range(training_iter):
        optimizer.zero_grad() 
        output = model(train_inputs)  # MultivariateNormal distribution
        loss = -mll(output, train_targets, M, M_H)  # loss function: minus log marginal likelihood 
        loss.backward() 

        print("Iter %d/%d - Loss: %.3f" %(
            i + 1, training_iter, 
            loss.item()
        ))

        optimizer.step() 
    

def model_prediction(model:GPModelWithHessians):
    '''
    Test the prediction function of the model
    '''
    # Set the model and likelihood into eval() mode 
    likelihood = model.likelihood 
    likelihood.eval() 
    model.eval() 

    # Test points 
    test_data_number = 10 
    training_inputs = model.train_inputs[0] 
    ndofs = training_inputs.shape[-1]
    test_data_hessian_data_point_index = torch.tensor([0, 2, 4, 8])
    
    # generate inputs and targets for the test set.
    test_inputs, test_targets = prepare_test_data(test_data_number, test_data_hessian_data_point_index)

    fixdofs = model.hessian_fixdofs 
    test_M_H = len(test_data_hessian_data_point_index)

    # Make predictions 
    with torch.no_grad():
        prediction_latent_function = model(test_inputs, inputs_hessian_data_point_index= test_data_hessian_data_point_index)
        predictions = likelihood(prediction_latent_function, test_data_number, test_M_H)

        test_prediction_mean = predictions.mean 

        pots, grads, hessians = transform_1d_train_targets_into_pots_grads_hessians(test_prediction_mean, test_data_number, 
                                                                                    ndofs, fixdofs, test_M_H)

    return test_inputs, test_targets, pots, grads, hessians 

def add_training_inputs(model:GPModelWithHessians):
    '''
    Add more data points into the training data.
    '''
    ndof = 2 
    hessian_fixdofs = torch.tensor([])
    nactive = int(ndof - len(hessian_fixdofs))
    hessian_triu_size = int((nactive + 1) * nactive / 2)

    xv, yv = torch.meshgrid(torch.linspace(0,1,10) + 0.05 , torch.linspace(0,1,10) + 0.05, indexing = 'ij' )
    
    M = xv.numel()
    new_hessian_data_point_index = torch.arange(0, 100, 5)
    
    new_train_inputs = torch.cat(
        (
        xv.contiguous().view(xv.numel(), 1),
        yv.contiguous().view(yv.numel(), 1)
        ),
        dim = 1
    )
    train_x_with_hessian = torch.index_select(new_train_inputs, dim= 0, index= new_hessian_data_point_index)
    f, gradient = franke(new_train_inputs[:, 0], new_train_inputs[:, 1])
    hessian = franke_hessian(train_x_with_hessian[:, 0], train_x_with_hessian[:, 1])
    
    # add noise to the function, gradient and hessian.  The value here should match perpare_likelihood_noise()
    pot_noise = np.power(10.0, -3)
    force_noise = np.power(10.0, -2)
    hessian_noise = np.power(10.0, -2)

    f = f + pot_noise * torch.rand(len(f))
    gradient = gradient + force_noise * torch.rand(len(gradient))
    hessian = hessian + hessian_noise * torch.rand(len(hessian))

    target_len = int(M * (ndof + 1) + hessian_triu_size * len(new_hessian_data_point_index))
    new_train_targets  = torch.cat( (f, gradient, hessian) , dim= 0 )
    assert len(new_train_targets) == target_len, "the length of target data is wrong."

    RBFHessian_gp.update_model_with_new_data(model, new_train_inputs, new_train_targets, new_hessian_data_point_index)
    

def test_RBFHessianGP():
    # generate input data 
    train_inputs, train_targets, hessian_fixdofs, hessian_data_point_index = prepare_training_inputs()
    M, ndof = train_inputs.shape
    nactive = ndof - len(hessian_fixdofs)
    hessian_triu_size = int(nactive * (nactive + 1) / 2) 

    gpr_SE_kernel_number = 1 
    
    # generate variance of noise:
    pot_noise, force_noise, hessian_noise = prepare_likelihood_noise(ndof, hessian_triu_size)

    # generate outputscale and lengthscale for the kernel
    kernel_outputscale, kernel_length_scale_ratio = prepare_kernel_prior(ndof, gpr_SE_kernel_number)
    
    # create Gaussian Process Regression model
    gp_model = GPModelWithHessians(train_inputs, train_targets,
                                   hessian_data_point_index, hessian_fixdofs,
                                    gpr_SE_kernel_number, 
                                    kernel_outputscale, kernel_length_scale_ratio,
                                    pot_noise, force_noise, hessian_noise)
    
    # train the model on the training data
    # model_training(gp_model, train_inputs, train_targets)
    RBFHessian_gp.train_gpr_model(gp_model)

    # test the model prediction with the test data.
    test_inputs, test_targets, pots, grads, hessians  = model_prediction(gp_model)

    # add more data points to the model.
    add_training_inputs(gp_model)

    # test the model prediction with the test data.
    new_test_inputs, new_test_targets, new_pots, new_grads, new_hessians  = model_prediction(gp_model)


    pass 

test_RBFHessianGP()