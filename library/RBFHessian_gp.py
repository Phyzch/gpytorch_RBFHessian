from .RBFHessianKernel import RBFKernelHessian 
from .RBFHessianMean import ConstantMeanHessian 
from .RBFHessian_prediction_strategy import RBFHessianPredictionStrategy
from .RBFHessian_gaussian_likelihood import RBFHessianGaussianLikelihood
import torch 
import gpytorch 
import numpy as np 
from gpytorch import settings 
from gpytorch.utils.generic import length_safe_zip
from gpytorch.distributions import MultivariateNormal
import warnings 
from gpytorch.utils.warnings import GPInputWarning

class GPModelWithHessians(gpytorch.models.ExactGP):
    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor, 
                 training_data_hessian_data_point_index: torch.Tensor, hessian_fixdofs: torch.Tensor,
                 gpr_SE_kernel_number : int,
                 kernel_outputscale: np.ndarray, kernel_lengthscale_ratio: np.ndarray,
                 likelihood_pot_noise: np.ndarray, likelihood_force_noise: np.ndarray, likelihood_hessian_noise: np.ndarray):
        '''
        :param: gpr_SE_kernel_number: number of squared exponential kernel for GPR model.
        :param: kernel_outputscale: numpy array, shape: [gpr_SE_kernel_number]   Estimation of the output scale of kernel
        :param: kernel_lengthscale_ratio: numpy array. shape: [gpr_SE_kernel_number, ard_num_dims]   Estimation of the length scale of kernel.
        :param: likelihood_pot_noise: numpy array. shape: [1]. Estimation of the variance of the potential noise.
        :param: likelihood_force_noise: numpy array. shape: [ard_num_dims]: Estimation of the variance of the force noise.
        :param: likelihood_hessian_noise: numpy array. shape: [hessian_triu_size]. Estimation of the variance of the hessian noise. 
        '''
        # the data point index that contains the hessian information.
        self.training_data_hessian_data_point_index = training_data_hessian_data_point_index

        self.hessian_fixdofs = hessian_fixdofs  # dofs that we will not include in hessian calculations.
        ard_num_dims = train_inputs.shape[-1]
        data_num = train_inputs.shape[-2]

        self.ard_num_dims = ard_num_dims 

        nactive = ard_num_dims - len(hessian_fixdofs)
        hessian_triu_size = int(nactive * (nactive + 1) / 2)

        target_len = data_num * (ard_num_dims + 1) + hessian_triu_size  * len(training_data_hessian_data_point_index)
        assert len(train_targets) == target_len, "the length of target data is wrong."
        
        # set the likelihood function 
        likelihood = self._set_likelihood_noise_prior(train_inputs, likelihood_pot_noise, likelihood_force_noise, likelihood_hessian_noise)

        super(GPModelWithHessians, self).__init__(train_inputs, train_targets, likelihood)

        # constraint for mean:
        self._set_mean_function(train_inputs, train_targets)

        # set the covariance function (kernel) for Gaussian Process regression.
        self._set_gpr_kernel(train_inputs, gpr_SE_kernel_number, kernel_outputscale, kernel_lengthscale_ratio)



    def _set_mean_function(self, train_inputs, train_targets):
        '''
        set the mean function for the Gaussian Process Regression.
        '''
        data_num = train_inputs.shape[-2]
        mean_constant_estimate = torch.mean(train_targets[..., :data_num], dim= -1)
        self.mean_module = ConstantMeanHessian()
        self.mean_module.constant = mean_constant_estimate  # set the constant (size 1) as mean value of prior 

    def _set_gpr_kernel(self, train_inputs, gpr_SE_kernel_number, kernel_outputscale, kernel_lengthscale_ratio):
        '''
        set the kernel for the Gaussian Process Regression 
        '''
        self.gpr_SE_kernel_number = gpr_SE_kernel_number 

        ard_num_dims = train_inputs.shape[-1]

        covar_module_component_list = []
        base_kernel_component_list = []

        # we choose Gamma distribution as prior distribution for output scale and length scale. 
        # See https://docs.gpytorch.ai/en/stable/priors.html  &  https://www.wikiwand.com/en/Gamma_distribution
        # alpha parameter (shape parameter) for the gamma distribution of the length and outputscale. 
        length_gamma_alpha = 3.0
        output_gamma_alpha = 3.0        

        for i in range(gpr_SE_kernel_number):
            # The prior distribution of the length scale of the parameter is decided by the initial training inputs (we only provides the ratio).
            # this is bad for cross validation, but for simply training model, it works fine.
            train_inputs_range = torch.max(train_inputs, dim= 0).values - torch.min(train_inputs , dim= 0).values 
            length_scale = torch.from_numpy(kernel_lengthscale_ratio[i]) * train_inputs_range 
            length_gamma_beta = torch.div(length_gamma_alpha, length_scale)

            output_scale = kernel_outputscale[i]

            # set prior for lengthscale and outputscale 
            lengthscale_prior = gpytorch.priors.GammaPrior(length_gamma_alpha, length_gamma_beta) 
            outputscale_prior = gpytorch.priors.GammaPrior(output_gamma_alpha, output_gamma_alpha / output_scale) 

            # add lengthscale constraint
            length_scale_ratio_cutoff = 0.1 
            length_scale_cutoff = length_scale_ratio_cutoff * train_inputs_range 
            lengthscale_constraint = gpytorch.constraints.GreaterThan(length_scale_cutoff) 

            # set Squared exponential kernel function 
            base_kernel = RBFKernelHessian(ard_num_dims= ard_num_dims, 
                                            lengthscale_prior= lengthscale_prior, 
                                            lengthscale_constraint= lengthscale_constraint,
                                            hessian_fixdofs= self.hessian_fixdofs
                                            )
            
            covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior = outputscale_prior)

            # Initialize lengthscale and output scale to the mean of priors
            covar_module.base_kernel.lengthscale = lengthscale_prior.mean 
            covar_module.outputscale = outputscale_prior.mean 

            base_kernel_component_list.append(base_kernel)
            covar_module_component_list.append(covar_module)

        self.base_kernel_component_list = base_kernel_component_list 
        self.covar_module_component_list = covar_module_component_list

        # sum of Squared Exponential Covariance function. 
        self.covar_module = self.covar_module_component_list[0]
        for i in range(1, gpr_SE_kernel_number):
            self.covar_module = self.covar_module + self.covar_module_component_list[i]

    def _set_likelihood_noise_prior(self, train_inputs,
                                    likelihood_pot_noise, likelihood_force_noise, likelihood_hessian_noise):
        '''
        set the prior and constraint for the likelihood noise.
        '''
        ard_num_dims = train_inputs.shape[-1]
        data_point_nums = train_inputs.shape[-2]
        batch_shape = train_inputs.shape[:-2]

        nactive = ard_num_dims - len(self.hessian_fixdofs)
        hessian_triu_size = int((nactive + 1) * nactive / 2)

        # First: check the shape of the potential noise and force noise 
        assert likelihood_pot_noise.shape[0] == 1, "the shape of potential noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(likelihood_pot_noise.shape[0], 1)
        assert likelihood_force_noise.shape[0] == ard_num_dims, "the shape of the force noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(likelihood_force_noise.shape[0], ard_num_dims)
        assert likelihood_hessian_noise.shape[0] == hessian_triu_size, "the shape of hessian noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(likelihood_hessian_noise.shape[0], hessian_triu_size)

        likelihood_pot_noise_var = np.power(likelihood_pot_noise, 2)
        likelihood_force_noise_var = np.power(likelihood_force_noise, 2)
        likelihood_hessian_noise_var = np.power(likelihood_hessian_noise, 2)

        # pot noise prior and pot noise constraint
        pot_noise_mean = torch.from_numpy(likelihood_pot_noise_var)
        pot_noise_std = torch.from_numpy(likelihood_pot_noise_var / 10)
        pot_noise_prior = gpytorch.priors.NormalPrior(pot_noise_mean, pot_noise_std)

        pot_noise_lower_bound = pot_noise_mean.div(10)
        pot_noise_upper_bound = pot_noise_mean.mul(10)
        pot_noise_constraint = gpytorch.constraints.Interval(pot_noise_lower_bound, pot_noise_upper_bound)

        # force noise prior and force noise constraint:
        force_noise_mean = torch.from_numpy(likelihood_force_noise_var)
        force_noise_std = torch.from_numpy(likelihood_force_noise_var / 10) 
        force_noise_prior = gpytorch.priors.NormalPrior(force_noise_mean, force_noise_std)

        force_noise_lower_bound = force_noise_mean.div(10)
        force_noise_upper_bound = force_noise_mean.mul(10)
        force_noise_constraint = gpytorch.constraints.Interval(force_noise_lower_bound, force_noise_upper_bound)

        # hessian noise prior and noise constraint:
        hessian_noise_mean = torch.from_numpy(likelihood_hessian_noise_var) 
        hessian_noise_std = torch.from_numpy(likelihood_hessian_noise_var / 10) 
        hessian_noise_prior = gpytorch.priors.NormalPrior(hessian_noise_mean, hessian_noise_std)

        hessian_noise_lower_bound = hessian_noise_mean.div(10)
        hessian_noise_upper_bound = hessian_noise_mean.mul(10)
        hessian_noise_constraint = gpytorch.constraints.Interval(hessian_noise_lower_bound, hessian_noise_upper_bound)

        # likelihood function 
        likelihood = RBFHessianGaussianLikelihood(ard_num_dims, hessian_triu_size, batch_shape, 
                                                  pot_noise_prior, pot_noise_constraint,
                                                  force_noise_prior, force_noise_constraint,
                                                  hessian_noise_prior, hessian_noise_constraint)
        
        # set the initial value of pot noise, force noise and hessian noise
        likelihood.pot_noises = pot_noise_mean 
        likelihood.force_noises = force_noise_mean 
        likelihood.hessian_noises = hessian_noise_mean 

        return likelihood  

    def forward(self,x, inputs_hessian_data_point_index= torch.tensor([]), **kwargs):
        '''
        return the distribution of the training targets
        ''' 
        M_H = len(inputs_hessian_data_point_index)
        nactive = self.ard_num_dims - len(self.hessian_fixdofs)
        mean_x = self.mean_module(x, M_H= M_H, nactive= nactive)
        with settings.lazily_evaluate_kernels(False):
            covar_x = self.covar_module(x, x, hessian_data_point_index_1= inputs_hessian_data_point_index, hessian_data_point_index_2 = inputs_hessian_data_point_index)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, *args, **kwargs):
        '''
        *args are new input data (either training inputs or test inputs)
        **kwargs: key word arguments.
        :param: inputs_hessian_data_point_index is used for covariance matrix (kernel) evaluation.
        '''
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]  # make inputs data have 2 dimensions.

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

            res = gpytorch.module.Module.__call__(self, *inputs, inputs_hessian_data_point_index= self.training_data_hessian_data_point_index)   # this will call the forward() function. hessian_data_point_index is in **kwargs.
            return res 
        
        #Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args 
            
            inputs_hessian_data_point_index = kwargs.get("inputs_hessian_data_point_index")
            if inputs_hessian_data_point_index == None:
                raise RuntimeError("Must provide inputs_hessian_data_point_index for computing kernel.")
            if  type(inputs_hessian_data_point_index) != torch.Tensor:
                raise RuntimeError("The inputs_hessian_data_point_index must be a tensor.")
            
            full_output = gpytorch.module.Module.__call__(self, *full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("GPModelWithHessian.forward method must return a MultivariateNormal")
            
            return full_output 
        
        # Posterior mode: Compute the posterior prediction of the GPR model.
        else:
            inputs_hessian_data_point_index = kwargs.get("inputs_hessian_data_point_index")
            if inputs_hessian_data_point_index == None:
                raise RuntimeError("Must provide inputs_hessian_data_point_index for computing kernel.")
            if  type(inputs_hessian_data_point_index) != torch.Tensor:
                raise RuntimeError("The inputs_hessian_data_point_index must be a tensor.")
    
            if all(torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)):
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning
                )
            
            # make the prediction:
            # Get the terms that only depend on training data 
            if self.prediction_strategy is None:
                train_outputs = gpytorch.module.Module.__call__(self, *train_inputs, inputs_hessian_data_point_index= self.training_data_hessian_data_point_index)

                # Create the prediction strategy 
                self.prediction_strategy = RBFHessianPredictionStrategy(
                    train_inputs= train_inputs[0],
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
            inputs_hessian_data_point_index_in_full_input = inputs_hessian_data_point_index + train_inputs[0].shape[-2]
            full_inputs_hessian_data_point_index = torch.cat((self.training_data_hessian_data_point_index, inputs_hessian_data_point_index_in_full_input))

            full_output = gpytorch.module.Module.__call__(self, *full_inputs, inputs_hessian_data_point_index= full_inputs_hessian_data_point_index)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar, inputs_hessian_data_point_index, inputs[0].shape[-2] )

            return full_output.__class__(predictive_mean, predictive_covar)