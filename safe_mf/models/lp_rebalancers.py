import cvxpy as cp
import numpy as np
from scipy.optimize import linear_sum_assignment


class StaticRebalancer:
    def __init__(self, num_zones):
        self.n = num_zones
        self.build_variables()
        self.build_objective()
        self.build_constraints()
        self.problem = cp.Problem(cp.Minimize(self.obj),self.constraints)
    
    def build_variables(self):
        self.alpha = cp.Variable((self.n,self.n))
        # travel times
        #self.T = cp.Parameter((self.n,self.n),nonneg=True)
        #self.T = cp.Parameter(nonneg=True)
        # demand vector
        self.lamb = cp.Parameter((self.n))
        # self.lamb_out = cp.Parameter((self.n))
        # destination probability matrix, row origin, column destination
        self.P = cp.Parameter((self.n,self.n))
        return
    
    def build_objective(self):
        self.obj = cp.sum(self.alpha)
        return
    
    def build_constraints(self):
        self.constraints = []
        for i in range(self.n):
            self.constraints = (self.constraints +
                [cp.sum(self.alpha[i,:])-cp.sum(self.alpha[:,i]) == 
                 -self.lamb[i]+cp.sum(cp.multiply(self.lamb,self.P[:,i]))])
            # self-loop always zero
            # self.constraints = self.constraints + [self.alpha[i,i] == 0]
            # total rebalancing from one cell can't exceed the excessive supply
            # self.constraints = self.constraints + [cp.sum(self.alpha[i,:]) <= self.excessive[i]]
        self.constraints = self.constraints + [self.alpha >= 0]
        return


class RealTimeRebalancer:
    def __init__(self, num_zones):
        self.n = num_zones
        self.build_variables()
        self.build_objective()
        self.build_constraints()
        self.problem = cp.Problem(cp.Minimize(self.obj),self.constraints)

    def build_variables(self):
        self.num = cp.Variable((self.n,self.n))
        # travel times
        # self.T = cp.Parameter((self.n,self.n),nonneg=True)
        #self.T = cp.Parameter(nonneg=True)
        # excessive cars
        self.v_ex = cp.Parameter((self.n))
        # desired cars
        self.v_d = cp.Parameter((self.n))
        return
    
    def build_objective(self):
        self.obj = cp.sum(self.num)
        return
    
    def build_constraints(self):
        self.constraints = []
        for i in range(self.n):
            # the updated number of vehicles should be greater then the desired number of vehicles
            self.constraints = (self.constraints +
                [self.v_ex[i]+cp.sum(self.num[:,i])-cp.sum(self.num[i,:]) >= self.v_d[i]])
            # self-loop always zero
            # self.constraints = self.constraints + [self.num[i,i] == 0]
            # total rebalancing flows from one cell can't exceed the excessive supply
            # self.constraints = self.constraints + [cp.sum(self.num[i,:]) <= self.v_ex[i]]
        self.constraints = self.constraints + [self.num >= 0]