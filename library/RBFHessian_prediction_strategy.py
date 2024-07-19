import torch
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch import settings 

class RBFHessianPredictionStrategy(DefaultPredictionStrategy):
    '''
    prediction strategy for the kernel which is Radial Basis Function, and the case we have the Hessian data in the target data.
    '''
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, training_data_hessian_data_point_index, hessian_fixdofs):
        '''
        :param: train_inputs: training input data.
        :param: train_prior_dist: Prior distribution of the training targets (MultivariateNormal Distribution)
        :param: train_labels: training targets data (y)
        :param: likelihood: likelihood function, add noise to the GPR model.
        :param: training_data_hessian_data_point_index: the index in the training data that has the hessian information
        :param: hessian_fixdofs: the index of dofs we exclude from the hessian calculation.
        '''
        self.train_inputs = train_inputs  # input X  
        self.train_prior_dist = train_prior_dist  # prior distribution f(X)
        self.train_labels = train_labels # targets (train_y)
        self.likelihood = likelihood   # the likelihood here is the RBFHessianGaussianLikelihood instance.

        M_H = len(training_data_hessian_data_point_index)
        M = train_inputs.shape[-2]

        mvn = self.likelihood(train_prior_dist, M, M_H)  # probability distribution of trained_y 
        self.lik_train_train_covar = mvn.lazy_covariance_matrix  # K(X,X) + sigma^2 I 

        self.training_data_hessian_data_point_index = training_data_hessian_data_point_index 
        self.hessian_fixdofs = hessian_fixdofs 

        self.train_num = train_inputs.shape()[-2]  # number of training points 
        self.d = train_inputs.shape()[-1]   # dimension d of inputs 
        self.nactive = self.d - len(hessian_fixdofs)  # number of active dofs
        self.hessian_triu_size = (self.nactive + 1) * self.nactive / 2  # the size of upper triangle part of the hessian.

    def exact_prediction(self, joint_mean, joint_covar, test_data_hessian_data_point_index, test_data_num):
        '''
        Compute the posterior predictive mean and covariance of a GP.
        :param: joint_mean:  mean value of joint gaussian distribution of training data and test data.
        :param: joint_covar: covariance matrix of joint gaussian distribution of training data and test data.
        :param: test_data_hessian_data_point_index: data index in the test data that has hessian information.
        :param: test_data_num: number of test data points 
        '''  
        MH_1 = len(self.training_data_hessian_data_point_index)  # number of training data that contains hessian information.
        MH_2 = len(test_data_hessian_data_point_index)  # number of test data that contains hessian information.

        M1 = self.train_num  # number of training data
        M2 = test_data_num   # number of test data 

        d = self.d # dimensions of the input data 
        hessian_triu_size = self.hessian_triu_size 

        training_target_pots_index = torch.arange(start= 0, end= M1)
        training_target_grads_index = torch.arange(start= (M1 + M2), end= (M1 + M2) + M1 * d)
        training_target_hessian_index = torch.arange(start= (M1 + M2) * (d + 1), end= (M1 + M2) * (d + 1) + MH_1 * hessian_triu_size)
        training_target_index = torch.concat((training_target_pots_index, training_target_grads_index, training_target_hessian_index),  dim= -1)

        test_target_pots_index = torch.arange(start= M1, end= M1 + M2)
        test_target_grads_index = torch.arange(start= (M1 + M2) + M1 * d, end= (M1+ M2) * (d + 1))
        test_target_hessian_index = torch.arange(start= (M1 + M2) * (d + 1) + MH_1 * hessian_triu_size, end= (M1 + M2) * (d + 1) + (MH_1 + MH_2) * hessian_triu_size)
        test_target_index = torch.concat( (test_target_pots_index, test_target_grads_index, test_target_hessian_index), dim= -1 )
        
        test_mean = torch.index_select(joint_mean, dim= -1, index= test_target_index)

        if joint_covar.size(-1) < settings.max_eager_kernel_size.value(): 
            test_covar = torch.index_select(joint_covar, dim= -2, index= test_target_index).to_dense()
        else:
            test_covar = torch.index_select(joint_covar, dim= -2, index= test_target_index)

        test_test_covar = torch.index_select(joint_covar, dim= -1, index= test_target_index)
        test_train_covar = torch.index_select(joint_covar, dim= -1, index= training_target_index)

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar)
        )
    