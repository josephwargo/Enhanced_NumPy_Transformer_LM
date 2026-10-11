import numpy as np
import cupy as cp
import Optimizer.scheduler as sch

class adamw_optimizer(object):
####################################
# Initializations #
####################################
    def __init__(self, model_dict, grad_dict, reg_factor, scheduler_type, eta_min, eta_max, time_max):

        # dicts to store pointers to params, gradients, and per-param child optimizers
        self.model_dict = model_dict
        self.grad_dict = grad_dict
        self.optimizer_dict = {}

        self.reg_factor = reg_factor
        
        # constants for adamw first and second moments
        self.beta1 = .9
        self.beta2 = .999
        self.epsilon = 1e-8

        # only implemented for cosine annealing thus far
        self.scheduler = sch.scheduler(scheduler_type=scheduler_type, eta_min=eta_min, eta_max=eta_max, time_max=time_max)

        # 
        self.clip_val_threshold = 1
        self.global_norm = 1

        self.time_step = 1 # time step counter as we loop through each instance of the optimizer

    def init_all_adamw(self, optimizer_parent):
        # initializing an individual instance of adam optimizer for each parameter in the model. this is because each param will need its own vd and md specific to its shape
        # So the best way to architect this is to have a single parent optimizer (adamw_optimizer) with children for each model param (adamw_instance)
        # note that this is also adding the respective gradients relevant to each optimizer
        for params_name, params in self.model_dict.items():
            self.optimizer_dict['adamw_'+params_name] = self.adamw_instance(optimizer_parent, params, self.grad_dict[params_name])
        # TODO: update this to also load in the grad_dict
    
    # calculate global norm across all gradients of params
    def get_global_norm(self):
        sum_of_squares = 0
        for grad_name, grad in self.grad_dict.items():
            sum_of_squares += cp.sum(grad**2)
        self.global_norm = cp.sqrt(sum_of_squares)

    def clip_all_adamw(self):
        # getting global norm first
        self.get_global_norm()
        for optimizer_name, optimizer in self.optimizer_dict.items(): # looping through dict of child optimizers and calling function that clips grads
            optimizer.clip_grad()

    def update_all_adamw(self, learning_rate): # looping through dict of child optimizers and calling function that updates params
        for optimizer_name, optimizer in self.optimizer_dict.items():
            optimizer.update_adamw(learning_rate)
        self.time_step += 1

    # the adamw child instances that are unique for each param in the model
    class adamw_instance(object):
        def __init__(self, optimizer_parent, params, grads):
            self.params = params
            self.optimizer_parent = optimizer_parent
            self.grads=grads
            # momentum
            self.md_params = cp.zeros(shape=params.shape).astype(cp.float32)
            # variance
            self.vd_params = cp.zeros(shape=params.shape).astype(cp.float32)

        # clipping by the global norm, but passing in parameter from model and doing the update inplace (no duplicates, saves memory)
        def clip_grad(self):
            if self.optimizer_parent.global_norm > self.optimizer_parent.clip_val_threshold:
                self.grads *= self.optimizer_parent.clip_val_threshold / self.optimizer_parent.global_norm

        def update_adamw(self, learning_rate):
            # momentum stored
            self.md_params *= self.optimizer_parent.beta1
            self.md_params += (1-self.optimizer_parent.beta1)*self.grads
            
            # variance stored
            self.vd_params *= self.optimizer_parent.beta2
            self.vd_params += (1-self.optimizer_parent.beta2)*(self.grads**2)

            # non-persistent updates to momentum and variance that are specific to this update 
            # doing ^t on beta1 and beta2 once per step
            b1T = self.optimizer_parent.beta1**self.optimizer_parent.time_step
            b2T = self.optimizer_parent.beta2**self.optimizer_parent.time_step
            md_params_hat = self.md_params/(1-b1T)
            vd_params_hat = self.vd_params/(1-b2T)

            schedule_multiplier = self.optimizer_parent.scheduler.get_lr(self.optimizer_parent.time_step)
            print(f'Schedule multiplier val: {schedule_multiplier}')

            # calculating update val in multiple steps so it's explainable
            decoupled_weight_decay = self.optimizer_parent.reg_factor * self.grads
            pre_scheduled_update = (learning_rate * md_params_hat) / (cp.sqrt(vd_params_hat) + self.optimizer_parent.epsilon) + decoupled_weight_decay
            
            
            update_val = schedule_multiplier*pre_scheduled_update
            
            # print(f'Pre-Scheduled update val shape: {pre_scheduled_update.shape}')
            # print(f'Param shape: {self.params.shape}')
            # print(f'Pre-Scheduled update val sum: {cp.sum(pre_scheduled_update)}')
            # print(f'Update val shape: {update_val.shape}')
            # print()


            self.params += -update_val

        
# 11.1715726852417