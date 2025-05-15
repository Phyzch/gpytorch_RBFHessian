import numpy as np 
import torch 
import gpytorch 
from library.RBFHessian_gp import GPModelWithHessians 
from library.RBFHessian_utils import transform_1d_train_targets_into_pots_grads_hessians
import library.RBFHessian_gp as RBFHessian_gp
import time 
from functools import wraps 

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time 
        print(f"Function '{func.__name__}' executed in {duration:.6f} seconds.")
        return result 
    return wrapper

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
    test_inputs = torch.rand(test_data_number, ndofs).to(device= test_data_hessian_data_point_index.device)
    test_inputs_with_hessian = torch.index_select(test_inputs, dim= 0, index= test_data_hessian_data_point_index)

    f, gradient = franke(test_inputs[:, 0], test_inputs[:, 1])
    hessian = franke_hessian(test_inputs_with_hessian[:, 0], test_inputs_with_hessian[:, 1])

    test_targets = torch.cat((f, gradient, hessian), dim= 0)

    return test_inputs, test_targets



@timeit
def model_training(model:GPModelWithHessians, train_inputs: torch.Tensor, train_targets: torch.Tensor,  training_error_cutoff= np.power(10.0, -3)):
    training_iter = 1000 
    likelihood = model.likelihood 

    # Find optimal model hyper-parameter 
    model.train() 
    likelihood.train() 

    # Use the adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.05)

    # "Loss" for GPS -- the marginal log likelihood 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    M = train_inputs.shape[-2]  # total number of data points.
    M_H = len(model.training_data_hessian_data_point_index)  # number of data points have hessian info

    # initialize loss_func_change and old_loss to enable while loop
    loss_func_change = 1000
    old_loss_value = 1000

    train_counts = 0 
    train_counts_output = 20

    while loss_func_change > training_error_cutoff:
        # reset the gradients of all optimized torch.Tensor 
        optimizer.zero_grad()   
        # output from model training data
        output = model(train_inputs)
        # calculate the loss function. here the returned loss is a torch.tensor.
        loss = - mll(output, train_targets, M, M_H)

        loss_value = loss.detach().item() 

        # calculate the change of loss function to decide whether we will stop the loop.
        loss_func_change = np.abs(loss_value - old_loss_value)
        old_loss_value = loss_value 

        # back propagation the loss function to compute the gradient of each parameter 
        loss.backward()
        
        # optimizer optimize the parameter using the gradient info.
        optimizer.step()

        train_counts = train_counts + 1 
        
        if train_counts % train_counts_output == 0:
            print("Iter %d - Loss: %.3f" %(
                train_counts, 
                loss.item())
            )
    
    print("Iter %d - Loss: %.3f" %(
                train_counts, 
                loss.item()))
    

def model_prediction(model:GPModelWithHessians):
    '''
    Test the prediction function of the model
    '''
    # Set the model and likelihood into eval() mode 
    cuda_available = torch.cuda.is_available()
    likelihood = model.likelihood 
    likelihood.eval() 
    model.eval() 

    # Test points 
    test_data_number = 10 
    training_inputs = model.train_inputs[0] 

    ndofs = training_inputs.shape[-1]
    test_data_hessian_data_point_index = torch.tensor([0, 2, 4, 8], device= training_inputs.device)
    
    # generate inputs and targets for the test set.
    test_inputs, test_targets = prepare_test_data(test_data_number, test_data_hessian_data_point_index)

    fixdofs = model.hessian_fixdofs 
    test_M_H = len(test_data_hessian_data_point_index)

    # Make predictions 
    with torch.no_grad():
        prediction_latent_function = model(test_inputs, inputs_hessian_data_point_index= test_data_hessian_data_point_index)
        predictions = likelihood(prediction_latent_function, test_data_number, test_M_H)

        test_prediction_mean = predictions.mean

        if cuda_available:
            test_prediction_mean = test_prediction_mean.cpu()
            test_inputs = test_inputs.cpu() 
            test_targets = test_targets.cpu() 

        pots, grads, hessians = transform_1d_train_targets_into_pots_grads_hessians(test_prediction_mean, test_data_number, 
                                                                                    ndofs, fixdofs, test_M_H)

    return test_inputs, test_targets, pots, grads, hessians 

def add_training_inputs(model:GPModelWithHessians):
    '''
    Add more data points into the training data.
    '''
    device = model.train_inputs[0].device
    
    ndof = 2 
    hessian_fixdofs = torch.tensor([])
    nactive = int(ndof - len(hessian_fixdofs))
    hessian_triu_size = int((nactive + 1) * nactive / 2)

    xv, yv = torch.meshgrid(torch.linspace(0,1,10) + 0.05 , torch.linspace(0,1,10) + 0.05, indexing = 'ij' )
    xv= xv.to(device= device)
    yv= yv.to(device= device)

    M = xv.numel()
    new_hessian_data_point_index = torch.arange(0, 100, 5).to(device= device)
    
    new_train_inputs = torch.cat(
        (
        xv.contiguous().view(xv.numel(), 1),
        yv.contiguous().view(yv.numel(), 1)
        ),
        dim = 1
    ).to(device= device)

    train_x_with_hessian = torch.index_select(new_train_inputs, dim= 0, index= new_hessian_data_point_index)
    f, gradient = franke(new_train_inputs[:, 0], new_train_inputs[:, 1])
    hessian = franke_hessian(train_x_with_hessian[:, 0], train_x_with_hessian[:, 1])
    
    # add noise to the function, gradient and hessian.  The value here should match perpare_likelihood_noise()
    pot_noise = np.power(10.0, -3)
    force_noise = np.power(10.0, -2)
    hessian_noise = np.power(10.0, -2)

    f = f + pot_noise * torch.rand(len(f), device= f.device)
    gradient = gradient + force_noise * torch.rand(len(gradient), device= gradient.device)
    hessian = hessian + hessian_noise * torch.rand(len(hessian), device= hessian.device)

    target_len = int(M * (ndof + 1) + hessian_triu_size * len(new_hessian_data_point_index))
    new_train_targets  = torch.cat( (f, gradient, hessian) , dim= 0 ).to(device= device)

    assert len(new_train_targets) == target_len, "the length of target data is wrong."

    RBFHessian_gp.update_model_with_new_data(model, new_train_inputs, new_train_targets, new_hessian_data_point_index)
    
def use_cuda(train_inputs, train_targets, hessian_data_point_index, hessian_fixdofs):
    """
    if cuda is available, move all model and tensors on cuda.
    assign variable on cuda locally in function will not affect the variable outside the scope.
    Therefore, we must return the variable.
    """
    # put model and data on CUDA if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. GPU is enabled.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. Running on CPU.")
    
    # put data and model on gpu.
    if cuda_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_inputs= train_inputs.to(device)
        train_targets = train_targets.to(device)
        hessian_data_point_index = hessian_data_point_index.to(device)
        hessian_fixdofs = hessian_fixdofs.to(device)
        print(f"device {device}")

    return train_inputs, train_targets, hessian_data_point_index, hessian_fixdofs


def test_RBFHessianGP():
    # set the seed for reproductibility
    torch.manual_seed(0)
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
    
    (train_inputs, train_targets, hessian_data_point_index, hessian_fixdofs) = use_cuda(train_inputs, 
                                                                                        train_targets,
                                                                                        hessian_data_point_index, 
                                                                                        hessian_fixdofs) 
    print(f"train inputs device {train_inputs.device}")

    # create Gaussian Process Regression model
    gp_model = GPModelWithHessians(train_inputs, train_targets,
                                   hessian_data_point_index, hessian_fixdofs,
                                    gpr_SE_kernel_number, 
                                    kernel_outputscale, kernel_length_scale_ratio,
                                    pot_noise, force_noise, hessian_noise)
    gp_model.to(device= train_inputs.device)

    # train the model on the training data
    model_training(gp_model, train_inputs, train_targets)
    # RBFHessian_gp.train_gpr_model(gp_model)

    # test the model prediction with the test data.
    test_inputs, test_targets, pots, grads, hessians  = model_prediction(gp_model)

    # add more data points to the model.
    add_training_inputs(gp_model)

    # test the model prediction with the test data.
    new_test_inputs, new_test_targets, new_predicted_pots, new_predicted_grads, new_predicted_hessians  = model_prediction(gp_model)

    print(f"test input {new_test_inputs}")
    print(f"test targets: {new_test_targets}")
    print(f"predicted V {new_predicted_pots}")
    print(f"predicted grad {new_predicted_grads}")
    print(f"predicted hessian {new_predicted_hessians}")
    pass 

test_RBFHessianGP()