import numpy as np 
import torch 
import gpytorch 
from .library.RBFHessian_gp import GPModelWithHessians 
from .library.GPWithHessian import transform_1d_train_targets_into_pots_grads_hessians

def prepare_likelihood_noise(ndof, hessian_triu_size):
    '''
    generate the noise data for potential, force and hessian
    '''
    pot_noise = np.array([np.power(10.0, -6)]) 
    force_noise = np.ones([ndof]) * np.power(10.0, -4)
    hessian_noise = np.ones([hessian_triu_size]) * np.power(10.0, -3)


    return pot_noise, force_noise, hessian_noise 

def prepare_kernel_prior(ndof, gpr_SE_kernel_number):
    '''
    generate the prior distribution of the kernel function
    '''
    kernel_outputscale = np.ones([gpr_SE_kernel_number]) * 0.04 
    kernel_length_scale_ratio = np.ones([gpr_SE_kernel_number, ndof]) * 0.3

    return kernel_outputscale, kernel_length_scale_ratio

def prepare_training_inputs():
    '''
    generate the input and targets of the training data.
    '''
    M = 10 # data number 
    hessian_data_point_index = torch.tensor([2,3, 5])
    
    ndof = 4 
    hessian_fixdofs = torch.tensor([1])
    nactive = ndof - len(hessian_fixdofs)
    hessian_triu_size = (nactive + 1) * nactive / 2 

    train_inputs = torch.tensor(np.random.random([M , ndof]))

    target_len = M * ndof + hessian_triu_size * len(hessian_data_point_index)
    train_targets  = torch.tensor(np.random.random([target_len]))

    return train_inputs, train_targets, hessian_fixdofs, hessian_data_point_index 

def model_training(model:GPModelWithHessians, train_inputs: torch.Tensor, train_targets: torch.Tensor):
    training_iter = 20 

    # Fine optimal model hyper-parameter 
    likelihood = model.likelihood 
    model.train() 
    likelihood.train() 

    # Use the adam optimizer 
    optimizer = torch.optim.Adam(model.parameters, lr= 0.05)

    # "Loss" for GPS -- the marginal log likelihood 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad() 
        output = model(train_inputs)  # MultivariateNormal distribution
        loss = -mll(output, train_targets)  # loss function: minus log marginal likelihood 
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
    test_data_number = 3 
    training_inputs = model.train_inputs 
    ndofs = training_inputs.shape[-1]
    test_inputs = torch.tensor(np.random.random([test_data_number, ndofs]))

    M = training_inputs.shape[-2]
    fixdofs = model.hessian_fixdofs 
    M_H = len(model.training_data_hessian_data_point_index)
    # Make predictions 
    with torch.no_grad():
        prediction_latent_function = model(test_inputs)
        predictions = likelihood(prediction_latent_function)

        mean = predictions.mean 

        pots, grads, hessians = transform_1d_train_targets_into_pots_grads_hessians(mean, M, ndofs, fixdofs, M_H)

    return pots, grads, hessians 

def test_RBFHessianGP():
    # generate input data 
    train_inputs, train_targets, hessian_fixdofs, hessian_data_point_index = prepare_training_inputs()
    M, ndof = train_inputs.shape()
    nactive = ndof - len(hessian_fixdofs)
    hessian_triu_size = nactive * (nactive + 1) / 2 

    gpr_SE_kernel_number = 1 
    
    # generate noise:
    pot_noise, force_noise, hessian_noise = prepare_likelihood_noise(ndof, hessian_triu_size)

    # generate outputscale and lengthscale for the kernel
    kernel_outputscale, kernel_length_scale_ratio = prepare_kernel_prior(ndof, gpr_SE_kernel_number)
    
    gp_model = GPModelWithHessians(train_inputs, train_targets,
                                   hessian_data_point_index, hessian_fixdofs,
                                    gpr_SE_kernel_number, 
                                    kernel_outputscale, kernel_length_scale_ratio,
                                    pot_noise, force_noise, hessian_noise)
    
    model_training(gp_model, train_inputs, train_targets)

    pots, grads, hessians  = model_prediction(gp_model)
    