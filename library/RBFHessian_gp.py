from .RBFHessianKernel import RBFKernelHessian 
from .RBFHessianMean import ConstantMeanHessian 
from .RBFHessian_prediction_strategy import RBFHessianPredictionStrategy
import torch 
import gpytorch 
import numpy as np 
from gpytorch import settings 
from gpytorch.utils.generic import length_safe_zip
from gpytorch.distributions import MultivariateNormal
import warnings 
from gpytorch.utils.warnings import GPInputWarning

class GPModelWithHessians(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, training_data_hessian_data_point_index, hessian_fixdofs = torch.tensor([])):
        super(GPModelWithHessians, self).__init__(train_x, train_y, likelihood)
        # the data point index that contains the hessian information.
        self.training_data_hessian_data_point_index = training_data_hessian_data_point_index

        self.hessian_fixdofs = hessian_fixdofs  # dofs that we will not include in hessian calculations.
        ard_num_dims = 4 

        # constraint for mean:
        mean_prior = gpytorch.priors.NormalPrior(0.3, 0.01)
        self.mean_module = ConstantMeanHessian(prior= mean_prior)
        self.mean_module.constant = torch.nn.Parameter(torch.ones(1) * mean_prior.loc)  # set the constant (size 1) as mean value of prior 

        #register prior for kernel's length scale
        gamma_alpha = 3.0
        prior_lengthscale = torch.tensor([0.2] * ard_num_dims)
        lengthscale_prior = gpytorch.priors.GammaPrior(gamma_alpha, gamma_alpha/prior_lengthscale)
        outputscale = 1
        outputscale_prior = gpytorch.priors.GammaPrior(3.0 , gamma_alpha/outputscale)

        self.base_kernel = RBFKernelHessian(ard_num_dims= ard_num_dims, lengthscale_prior= lengthscale_prior, hessian_fixdofs= hessian_fixdofs)

        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel, outputscale_prior= outputscale_prior)

        # initialize the lengthscale and output scale as the mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        self.covar_module.outputscale = outputscale_prior.mean 

    def forward(self,x, hessian_data_point_index= torch.tensor([]), **kwargs):
        '''
        return the distribution of the training targets
        ''' 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x, hessian_data_point_index_1= hessian_data_point_index, hessian_data_point_index2 = hessian_data_point_index)
        return gpytorch.distributions.Multivariatenormal(mean_x, covar_x)
    
    def __call__(self, *args, **kwargs):
        '''
        *args are new input data (either training inputs or test inputs)
        **kwargs: key word arguments.
        :param: hessian_data_point_index is used for covariance matrix (kernel) evaluation.
        '''
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]  # make inputs data have 2 dimensions.
        inputs_hessian_data_point_index = kwargs.get("hessian_data_point_index")

        if inputs_hessian_data_point_index == None:
            raise RuntimeError("Must provide inputs_hessian_data_point_index for computing kernel.")
        if  type(inputs_hessian_data_point_index) != torch.Tensor:
            raise RuntimeError("The inputs_hessian_data_point_index must be a tensor.")

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )

            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)
                ):
                    raise RuntimeError("You must train on the training inputs!")

                if not torch.equal(inputs_hessian_data_point_index, self.training_data_hessian_data_point_index):
                    raise RuntimeError("The hessian_data_point_index you provide must match the training data in the training mode!")
                        

            res = gpytorch.module.Module.__call__(*input, **kwargs)   # this will call the forward() function. hessian_data_point_index is in **kwargs.
            return res 
        
        #Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args 
            full_output = gpytorch.module.Module.__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("GPModelWithHessian.forward method must return a MultivariateNormal")
            
            return full_output 
        
        # Posterior mode: Compute the posterior prediction of the GPR model.
        else:
            if all(torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)):
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning
                )
            
            # make the prediction:
            # Get the terms that only depend on training data 
            if self.prediction_strategy is None:
                train_outputs = gpytorch.module.Module.__call__(*input, **kwargs)

                # Create the prediction strategy 
                RBFHessianPredictionStrategy(
                    train_inputs= train_inputs,
                    train_prior_dist= train_outputs, 
                    train_labels= self.train_targets, 
                    likelihood= self.likelihood,
                    training_data_hessian_data_point_index= self.training_data_hessian_data_point_index,
                    hessian_fixdofs= self.hessian_fixdofs
                )
            
            # Concatenate the training input and test input into one input for generating the joint distribution 
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in length_safe_zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training / test data 
            full_output = gpytorch.module.Module.__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            return full_output.__class__(predictive_mean, predictive_covar)