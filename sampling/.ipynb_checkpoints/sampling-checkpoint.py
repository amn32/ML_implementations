import time
import numpy              as np
import pandas             as pd
import scipy.stats        as stats
import scipy.spatial      as spatial
import seaborn            as seabornInstance 
import matplotlib.pyplot  as plt
import statsmodels.api    as sm

from sklearn.linear_model import LinearRegression
from sklearn              import metrics
from tqdm.notebook        import tqdm

class MH():
    
    '''Metropolis Hastings Sampling'''
    
    def __init__(self, p_star, q_star, num_vars, samples = 30000, burn = 0.05):
        
        '''
        Parameters
        ----------
        
        p_star: func
            Target distribution
        q_star: func
            Proposal distribution
        num_vars: int
            Number of variables to be sampled
        samples: int
            Number of samples to take (Default is 30000)
        burn: float
            Burn in proportion (Default is 0.05)
        '''
        
        self.p_star      = p_star
        self.q_star      = q_star
        self.num_samples = samples
        self.burn        = round(self.num_samples*0.05)
        self.num_vars    = num_vars
        
        self.samples     = []
        self.rejected    = []
        
        self.x           = np.zeros(self.num_vars)
        
    def sample(self, x_train, y_train):
        
        '''
        Parameters
        ----------
        
        x_train: ndarray
            Training x data
        y_train: ndarray
            Training y data
        '''
        
        for i in tqdm(range(self.num_samples), leave = False):

            xdash     = self.q_star(self.x)

            threshold = self.p_star(xdash, x_train, y_train) - self.p_star(self.x, x_train, y_train)
            
            if threshold >= 1:

                self.x  = xdash

            elif np.random.uniform() < threshold:

                self.x  = xdash

            else:

                self.rejected.append(xdash)

            self.samples.append(self.x)
            
        self.samples = np.array(self.samples)
        
        self.final   = (self.samples[self.burn:,:].cumsum(0)/np.arange(1, (self.num_samples - self.burn)+1)[:,None])
        
        self.estimates = self.final[-1,:]
        
        return self.final
            

class Hamiltonian:
    
    '''MCMC leveraging Hamiltonian dynamics for faster conversion'''
    
    def __init__(self, energy, grads, num_vars, R = 5000, L = 100, epsilon0 = 0.1, burn1 = 0.2):
        
        '''
        Parameters
        ----------
        
        energy: func
            Hamiltonian energy function
        grads: func
            Hamiltonian energy function gradients
        num_vars: int
            Number of variables to be sampled
        R: int
            Number of iterations of Hamiltonian trajectory (Default is 5000)
        L: int
            Number of samples per trajectory (Default is 100)
        epsilon0: float
            Step size in Hamiltonian trajectory (Default is 0.1)
        burn1: float
            Burn in proportion (Default is 0.2)
        '''
        
        self.energy      = energy
        self.grads       = grads
        self.R           = R
        self.L           = L
        self.epsilon0    = epsilon0
        self.burn        = int(self.R/10)
        self.burn1       = int(self.R*burn1)
        self.num_vars    = num_vars
        self.samples     = []
        self.num_rejects = 0
        self.x           = np.zeros(self.num_vars)
        
        
    def sample(self, x_train, y_train):
        
        '''
        Parameters
        ----------
        
        x_train: ndarray
            Training x data
        y_train: ndarray
            Training y data
        '''
        
        for n in tqdm(range(-self.burn, self.R), leave = False):
        
            epsilon   = self.epsilon0*(1.0 + 0.1*np.random.normal())

            current_q = self.x.copy()

            q         = current_q.copy()

            M         = q.size 

            p         = np.random.normal(size=M)  # independent standard normal variates

            current_p = p.copy()

            p        -= epsilon * self.grads(q, x_train, y_train) / 2

            for i in range(self.L):

                q += epsilon * p # Make a full step for the position

                if i != self.L-1: 

                    p -= epsilon * self.grads(q, x_train, y_train) # Make a full step for the momentum, except at end of trajectory

            p -= epsilon * self.grads(q, x_train, y_train) / 2 # Make a half step for momentum at the end.

            current_energy  = self.energy(current_q, x_train, y_train)
            proposed_energy = self.energy(q,  x_train, y_train)

            current_K       = np.sum(current_p**2) / 2
            proposed_K      = np.sum(p**2) / 2

            threshold       = current_energy-proposed_energy+current_K-proposed_K

            if np.random.uniform() < np.exp(threshold):

                reject = 0

            else:

                q = current_q  # reject

                reject = 1

            self.x   = q.copy()

            rej = reject

            if n >= 0:

                self.num_rejects += rej
                self.samples.append(self.x)
            
        print('Acceptance Rate:', (self.R-self.num_rejects)/self.R)
        
        self.samples = np.array(self.samples)
        
        self.final   = (self.samples[self.burn1:,:].cumsum(0)/np.arange(1, (self.samples.shape[0] - self.burn1)+1)[:,None])
        
        self.estimates = self.final[-1,:]
        
        return self.samples
        
