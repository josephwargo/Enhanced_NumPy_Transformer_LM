import numpy as np
import cupy as cp
import Optimizer.scheduler as sch

class adamw_optimizer(object):
####################################
# Initializations #
####################################
    def __init__(self, model_dict, reg_factor, scheduler_type, eta_min, eta_max, time_max):

        self.model_dict = model_dict

        self.reg_factor = reg_factor
        
        # constants
        self.beta1 = .9
        self.beta2 = .999
        self.epsilon = 1e-8
        # only implemented for cosine annealing thus far
        self.scheduler = sch.scheduler(scheduler_type=scheduler_type, eta_min=eta_min, eta_max=eta_max, time_max=time_max)

        self.optimizer_dict = {}

        self.clip_val = 1 #self.get_global_norm()

        # self.init_all_adamw()

    # TODO: calculate global norm for clip_val
    def get_global_norm(self):
        pass

    class adamw_instance(object):
        # def __init__(self, optimizer_parent, input_shape, output_shape):
        def __init__(self, optimizer_parent, params):
            self.params = params
            input_shape = params.shape[0]
            output_shape = params.shape[1]
            # parent - from which we inherit betas, epsilon, scheduler, and reg factor
            self.optimizer_parent = optimizer_parent
            # momentum
            self.md_params = cp.zeros(shape=(input_shape, output_shape)).astype(cp.float32)
            # variance
            self.vd_params = cp.zeros(shape=(input_shape, output_shape)).astype(cp.float32)

        # clipping by the global norm, but passing in parameter from model and doing the update inplace (no duplicates, saves memory)
        def clip_grad(self):
            cp.clip(self.params, -self.optimizer_parent.clip_val, self.optimizer_parent.clip_val, out=self.params)

        def update_adamw(self, time_step, learning_rate):
            # momentum stored
            self.md_params *= self.optimizer_parent.beta1
            self.md_params += (1-self.optimizer_parent.beta1)*self.params
            
            # variance stored
            self.vd_params *= self.optimizer_parent.beta2
            self.vd_params += (1-self.optimizer_parent.beta2)*(self.params**2)

            # non-persistent updates to momentum and variance that are specific to this update 
            # doing ^t on beta1 and beta2 once per step
            b1T = self.optimizer_parent.beta1**time_step
            b2T = self.optimizer_parent.beta2**time_step
            md_params_hat = self.md_params/(1-b1T)
            vd_params_hat = self.vd_params/(1-b2T)

            schedule_multiplier = self.optimizer_parent.scheduler.get_lr(time_step)

            # calculating update val in multiple steps so it's explainable
            decoupled_weight_decay = self.optimizer_parent.reg_factor * self.params
            pre_scheduled_update = (learning_rate * md_params_hat) / (cp.sqrt(vd_params_hat) + self.optimizer_parent.epsilon) + decoupled_weight_decay
            update_val = schedule_multiplier*pre_scheduled_update

            self.params += -update_val

        
    def init_all_adamw(self, optimizer_parent):
        for params_name, params in self.model_dict:
            self.optimizer_dict['adamw_'+params_name] = self.adamw_instance(optimizer_parent, params)

    def clip_all_adamw(self):
        for params_name, params in self.model_dict:
            params.clip_grad()

    def update_all_adamw(self, time_step, learning_rate):
        for params_name, params in self.model_dict:
            params.update_adamw(time_step, learning_rate)