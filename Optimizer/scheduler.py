import cupy as cp
from math import pi
class scheduler(object):
####################################
# Initializations #
####################################
    def __init__(self, scheduler_type, eta_min, eta_max, time_max):

        self.scheduler_type = scheduler_type # only implemented for cosine annealing thus far

        self.eta_min = eta_min
        self.eta_spread = eta_max - eta_min
        # self.time_step = time_step
        self.time_max = time_max


    def get_lr(self, time_step):
        time_step_ratio = pi * (time_step / self.time_max)
        eta_t = self.eta_min + (.5 * self.eta_spread * (1 + cp.cosine(time_step_ratio)))
        return eta_t
