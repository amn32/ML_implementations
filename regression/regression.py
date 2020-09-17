import numpy as np
import random
import pandas as pd
import os
import gc
import tensorflow.compat.v1 as tf
from matplotlib              import pyplot as plt
from tqdm.notebook           import tqdm
from pathos.multiprocessing  import Pool
from scipy.spatial.distance  import cdist
from scipy.spatial           import cKDTree
from numpy.linalg            import inv
from scipy.linalg            import cho_solve

tf.disable_eager_execution()
os.system("taskset -p 0xfffff %d" % os.getpid())

class _Base():
    
    '''Base class with useful error functions
    
    Methods
    -------
    
    mse(X, Y)
        Calculate the mean squared error
    batch_mse(X, Y, batch_size = 100)
        Calculate the mean squared error in batches
    rmse(X, Y)
        Calculate the root mean squared error
    '''
    
    def mse(self , X, Y):
        return np.mean(np.square(self(X) - Y))
    
    def batch_mse(self, X, Y, batch_size = 100):

        batches_X = np.array_split(X,batch_size)
        batches_Y = np.array_split(Y,batch_size)

        batch_errors = []

        for i in range(batch_size):
            
            batch_X = batches_X[i]
            batch_Y = batches_Y[i]
            batch_errors.append(np.mean(np.square(self(batch_X) - batch_Y)))
    
        return np.mean(batch_errors)
    
    def rmse(self , X, Y):
        return np.sqrt(np.mean(np.square(self(X) - Y)))

    
################################################################################################
    
class Regression(_Base):
    '''
    Simple (penalised) linear regression
    '''
    
    def __init__(self, intercept = False):
        
        '''
        Parameters
        ----------
        intercept: Bool
            Boolean to include a constant - not penalised (Default is True)
        '''
        
        self.intercept = intercept
    
    def fit(self, X, Y, p=0):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Training x data
        Y: ndarray
            Training y data
        p: float
            Value for penalisation term (Default is 0 - corresponding to OLS regression)
        '''
        
        identity = np.identity(X.shape[1])
        
        if self.intercept:
            
            X              = np.insert(X, 0, 1, 1)
            identity       = np.identity(X.shape[1])
            identity[0][0] = 0

        self.W = np.linalg.solve(X.T.dot(X) + p * identity, X.T.dot(Y))
        
        if self.intercept:
            
            self.b = self.W[0]
            self.W = self.W[1:]
            
        return self     
    
    def __call__(self, X):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Test x data
        '''
        
        if self.intercept:
        
            self.preds = X.dot(self.W) + self.b
        
        else:
            
            self.preds = X.dot(self.W)
            
        return self.preds.reshape(-1,1)

################################################################################################
    
class K_Neighbours_Regressor(_Base):
    
    '''K Nearest Neighbours Regressor, optimised with scipy KDTree'''
    
    def __init__(self, k = 5, p = 2, weight = 'distance'):
        
        '''
        Parameters
        ----------
        
        k: int
            Number of nearest neighbours to consider (Default is 5)
        p: int
            Order of norm - determines how distance to neighbour is calculated (Defualt is 2)
        weight: str
            Determines whether data points are weighted uniformly or proportionately to distance to neighbour. Options ['uniform', 'distance'] (Defualt is 'distance')
        
        '''
          
        super().__init__()
        self.k = k
        self.p = p
        self.weight = weight
    
    def fit(self, X, y):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Training x data
        Y: ndarray
            Training y data
        '''
        
        self.X = X.copy()
        self.y = y.copy()
        self.KDTree = cKDTree(X)
        self.distance, self.idx = self.KDTree.query(X, k = self.k, p = self.p, n_jobs = -1)
        return self
    
    def get_neighbours(self, X):

        distance, idx = self.KDTree.query(X, k = self.k, p = self.p, n_jobs = -1) 

        return idx, distance
    
    def __call__(self, X):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Test x data
        '''
        
        idx, distance = self.get_neighbours(X)
        
        if len(np.where(distance == 0)[0])>0:
            index_T = np.where(distance == 0)
            index_F = np.where(distance != 0)
            distance[index_T] = 1
            distance[index_F] = np.inf
            
        if self.weight == 'distance':
        
            self.weights   = np.divide((1/distance),((1/distance).sum(axis = 1, keepdims = True)))
        
        elif self.weight == 'uniform':
            
            self.weights   = np.full((X.shape[0], self.k), 1/self.k)

        return np.einsum('abc,ab->ac', self.y[idx,None], self.weights)

################################################################################################
    
class Decision_Tree(_Base):
    
    '''Simple Decision Tree for regression'''
    
    def __init__(self, max_depth = None, 
                 min_node_count  = 1, 
                 bins            = False, 
                 step            = 10):
        
        '''
        Parameters
        ----------
        
        max_depth: int
            Maximum depth of the decision tree (Default is None)
        min_node_count: int
            Minimum number of samples in node. Once this is hit, node is considered terminal (Default is 1)
        bins: int/Bool
            Data may be binned to speed up training. bins specifies the number of bins (Default is False)
        step: int
            Step size during training (Default is 10)
        '''
        
        self.max_depth = max_depth
        self.min_node_count = min_node_count
        self.step = step
        self.bins = bins
        random.seed(50)
        
    def fit(self, X, Y):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Training x data
        Y: ndarray
            Training y data
        '''
        
        if self.bins != False:
            
            X_ = np.zeros(X.shape)
            
            for i in range(X.shape[1]):
                x       = X[:,i].copy()
                x_min   = np.min(x)   
                x      -= x_min
                x_max   = np.max(x)
                x      /= x_max
                x      *= self.bins
                x       = np.round(x)
                x      /= self.bins
                x      *= x_max
                x      += x_min
                X_[:,i] = x
                
            self.tree = self.build_tree(X_,Y)
          
        else:
            
            self.tree = self.build_tree(X,Y)
            
        return self
        
    def get_split(self, X_train,Y_train):
        
        if len(np.unique(Y_train))>3:
        
            best_obj = np.inf

            for i in range(X_train.shape[1]):

                vals = np.unique(X_train[:,i])[:-1]

                for j in vals[::self.step]:

                    left_index  = X_train[:,i]<=j

                    right_index = X_train[:,i]>j

                    left_var    = np.std(Y_train[left_index])

                    right_var   = np.std(Y_train[right_index])

                    obj         = left_index.sum()*left_var + right_index.sum()*right_var

                    if obj < best_obj:
                        left_indices  = left_index
                        right_indices = right_index
                        best_value    = j
                        best_feature  = i
                        best_obj      = obj
            try:
                      
                return {'feature' : best_feature,
                    'split' : best_value,
                    'var' : best_obj, 
                    'left_indices' : left_indices,
                    'right_indices' : right_indices}
            except:
                
                pass

    
    def build_tree(self, x, y, depth=0):
        
        split = self.get_split(x, y)

        if depth==self.max_depth  or len(np.unique(y)) < self.min_node_count or split == None:
            
            return {'leaf' : True, 'values' : y.mean()}

        else:
            
            left = self.build_tree(x[split['left_indices'],:], y[split['left_indices']], depth + 1)
            right = self.build_tree(x[split['right_indices'],:], y[split['right_indices']], depth + 1)
            
            return {'leaf'    : False,
                    'feature' : split['feature'],
                    'split'   : split['split'],
                    'var'     : split['var'],
                    'left'    : left,
                    'right'   : right}
        
    def map_one_exemplar(self, sample):

        tree = self.tree
        
        while not tree['leaf']:
            
            if sample[tree['feature']] <= tree['split']:
                tree  = tree['left']
            else:
                tree = tree['right']
            
        return tree['values']
            
    def __call__(self, X):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Test x data
        '''
        
        return np.array(list(map(self.map_one_exemplar, X))).reshape(-1,1)
    
class Random_Forest(_Base):
    
    '''Parrelelised implementation of the above decision tree regression model'''
    
    def __init__(self, n_trees  = 4, 
                 weight         = 'OOB', 
                 bag_ratio      = 0.7, 
                 feature_ratio  = 0.7, 
                 min_node_count = 3, 
                 bins           = False, 
                 max_depth      = None, 
                 step           = 10, 
                 threads        = 4):
        
        '''
        Parameters
        ----------
        
        n_trees: int
            How many trees to use in the forest (Default is 4)
        weight: str
            How to weight trees, by Out of Bag Error or Uniformly. Options ['OOB', 'Uniform'] (Default is 'OOB')
        bag_ratio: float between 0 and 1
            Determines how many samples to use, remainders can be used for Out of Bag weighting (Default is 0.7)
        feature_ratio: float between 0 and 1
            Determines proportion of features randomly selected by each tree (Default is 0.7)
        min_node_count: int
            Minimum number of samples in node. Once this is hit, node is considered terminal (Default is 1)
        bins: int/Bool
            Data may be binned to speed up training. bins specifies the number of bins (Default is False)
        max_depth: int
            Maximum depth of the decision tree (Default is None)
        step: int
            Step size during training (Default is 10)
        threads: int
            How many threads to use in the multiprocessing (Default is 4)
        '''
        
        self.max_depth      = max_depth
        self.bag_ratio      = bag_ratio
        self.feature_ratio  = feature_ratio
        self.n_trees        = n_trees
        self.weight         = weight
        self.threads        = threads
        self.min_node_count = min_node_count
        self.bins           = bins
        self.step           = step
        
    def fit(self, X, Y):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Training x data
        Y: ndarray
            Training y data
        '''
        
        T    = self.n_trees
        FR   = self.feature_ratio
        BR   = self.bag_ratio
        MD   = self.max_depth
        STEP = self.step
        BINS = self.bins
        
        with Pool(processes = 16) as pool:
    
            np.random.seed(50)
    
            genfeatures = lambda i : np.sort(np.random.choice(range(X.shape[1]), size = int(X.shape[1] * FR), replace=False))

            genrows = lambda i : np.random.choice(range(X.shape[0]), size = int(X.shape[0] * BR), replace = True)

            features = list(map(genfeatures, range(T)))
            rows     = list(map(genrows, range(T)))

            build_tree = lambda i : Decision_Tree(max_depth = MD, bins = BINS, step = STEP).fit(X[rows[i]][:,features[i]],Y[rows[i]])

            trees = pool.map(build_tree,range(T))
            
        self.trees = trees
        
        self.features = features
        
        self.rows = rows
        
        if self.weight == 'OOB':
            
            OOB_error = []
            
            for i in range(T):
                
                index_rows = list(set(range(X.shape[0])) - set(rows[i]))
                
                index_cols = list(features[i])
                
                prediction = trees[i](X[index_rows][:,index_cols])
                
                diff = prediction - Y[index_rows]
                
                error = np.mean(np.square(diff))

                OOB_error.append(1/error)
        
            self.weights = np.divide(OOB_error,sum(OOB_error))
            
        elif self.weight == 'Uniform':
            
            self.weights = np.full(self.n_trees, 1/self.n_trees)
        
        return self
        
    def __call__(self, X):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Test x data
        '''
        
        trees = self.trees

        features = self.features

        n_trees = self.n_trees

        predict = lambda i : trees[i](X[:,features[i]])

        predictions = list(map(predict, range(n_trees)))
        
        return np.sum(predictions*self.weights[:,None,None],axis=0)

################################################################################################    
    
class Gaussian_Process(_Base):
    
    '''Numpy Gaussian process with simple gradient descent (using momentum).'''
    
    def __init__(self, kernel = 'Gaussian', seed = 0):
        
        '''
        Parameters
        ----------
        
        kernel: str
            Which kernel to use [Currently only supports 'Gaussian'] (Default is 'Gaussian')
        seed: int
            Specify random seed
        '''
        
        kernels          = {'Gaussian': self.Gaussian_Kernel}
        
        self.kernel_name = kernel
        self.kernel      = kernels[kernel]
        self.seed        = seed
    
    def fit(self, X, Y, alpha = 1, epochs = 100, momentum = 0.8, prior_c = 1):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Training x data
        Y: ndarray
            Training y data
        alpha: float
            Parameter to scale the gradient (Default is 1)
        epochs: int
            Number of training iterations (Default is 100)
        momentum:
            Parameter to scale the momentum term for convergence (Default is 0.8)
        prior_c:
            Prior on the constant terms in the likelihood (Default is 1)
        '''
        
        np.random.seed(self.seed)
        
        self.param_l    = np.random.normal(0,0.5)
        self.param_s    = np.random.normal(0,0.5)
        self.param_sig2 = np.random.normal(0,0.5)
        
        self.l          = np.exp(self.param_l)
        self.s          = np.exp(2*self.param_s)
        self.sig2       = np.exp(self.param_sig2)
        
        self.momentum   = momentum
        self.prior_c    = prior_c
        
        self.n          = len(X)
        alpha          /= self.n

        self.X          = X
        self.Y          = Y.reshape(-1,1)
        
        self.f_mu       = Regression().fit(X,Y)
        
        self.Ymu        = self.f_mu(X)
        self.Ys2        = np.std((self.Y - self.Ymu))
        
        self.Y_transform = (self.Y-self.Ymu)/self.Ys2
        
        self.I          = np.eye(self.n)
        
        self.YY_T       = self.Y_transform @ self.Y_transform.T
        
        self.grad_descent(alpha, epochs)
        
        return self
    
    def Gaussian_Kernel(self, X1, X2):
    
        self.norm  = cdist(X1, X2, metric = 'sqeuclidean')

        return  (self.s**2)*np.exp(-self.norm/(2*self.l**2))
    
    def grad_descent(self, alpha, epochs):
        
        self.LL        = np.zeros(epochs)
        
        grad_sig2 = 0
        grad_l    = 0
        grad_s    = 0
        
        for i in tqdm(range(epochs), leave = False):
        
            K                = self.kernel(self.X,self.X) + self.sig2 * self.I
            L                = np.linalg.cholesky(K)
            self.inv         = cho_solve((L,True), self.I)
            k_inv_y          = self.inv @ self.Y_transform
            K_inv_K_inv_T    = k_inv_y  @ k_inv_y.T

            grad_sig2       *= self.momentum
            grad_l          *= self.momentum
            grad_s          *= self.momentum
            
            ds               = 2*self.s * np.exp(-self.norm/(2*self.l**2))
            dl               = self.s**2 * np.exp(-self.norm/(2*self.l**2)) * (self.norm/self.l**3)
            dsig2            = self.sig2 * self.I
            
            grad_sig2       += 0.5*np.trace((K_inv_K_inv_T - self.inv) @ dsig2)
 
            grad_l          += 0.5*np.trace((K_inv_K_inv_T - self.inv) @ dl)

            grad_s          += 0.5*np.trace((K_inv_K_inv_T - self.inv) @ ds)
            
            grad_sig2       -= 0.5 * self.prior_c * self.param_sig2 
            grad_l          -= 0.5 * self.prior_c * self.param_l 
            grad_s          -= 0.5 * self.prior_c * self.param_s

            self.param_sig2 += alpha * grad_sig2
            self.param_l    += alpha * grad_l 
            self.param_s    += alpha * grad_s
         
            self.l          = np.exp(self.param_l)
            self.s          = np.exp(2*self.param_s)
            self.sig2       = np.exp(self.param_sig2)
            
            self.LL[i]      = -0.5 * (self.n*np.log(2*np.pi) + np.prod(np.linalg.slogdet(self.kernel(self.X,self.X) + self.sig2 * self.I))\
                                + np.trace(self.inv @ self.YY_T))
        
        K                = self.kernel(self.X,self.X) + self.sig2 * self.I
        L                = np.linalg.cholesky(K)

        self.inv         = cho_solve((L,True), self.I)           
    
    def p_Y_star(self, X, cov):

        K_Xstar_X       = self.kernel(X, self.X)
        
        self.mu         = K_Xstar_X @ self.inv @ self.Y_transform #What dimension should this be?
        
        if cov:
            self.var        = np.diag(self.kernel(X, X))                         # K**
            self.var        = self.var + np.diag(K_Xstar_X @ self.inv @ K_Xstar_X.T)        # K*t Kinv Kt*         
    
    def __call__(self, X, cov = False):
        
        '''
        Parameters
        ----------
        
        X: ndarray
            Test x data
        '''
        
        self.p_Y_star(X, cov)
        
        if not cov:
            return self.mu * self.Ys2 + self.f_mu(X)
        
        return self.mu * self.Ys2 + self.f_mu(X), self.var * self.Ys2 ** 2
    
class tfGP:
    
    '''Gaussian process implemented in Tensorflow:'''
    
    def __init__(self, distribution = 'Gaussian', 
                 process_y  = True,
                 variable_l = False,
                 dtype      = tf.float64, 
                 jitter     = 1.0e-8):
        
        '''
        Parameters
        ----------
        
        distribution: str
            Likelihood distribution. Options ['Gaussian', 'Poisson'] (Default is 'Gaussian')
        process_y: Bool
            Determines whether y will be preprocessed by taking the residuals of a linear regression (Default is True)
        variable_l:
            Use vector lengthscale to penalise each feature differently
        dtype: (Default is float 64)
        jitter: float
            Helps with matrix inversion (Default is 1e-08)
        '''
    
        self.variable_l   = variable_l
        self.process_y    = process_y
        self.dtype        = dtype
        self.jitter       = jitter # To aid with matrix inversion
        self.distribution = distribution
        
    def init_variable(self, value, positive = False, multi = False):
        
        if multi:
            
            if positive:

                assert value[0] > 0.0
                
#                 return tf.Variable(value, shape = (self.M,), dtype=self.dtype)
            
                return tf.exp(tf.Variable(np.log(value), shape = (self.M,), dtype = self.dtype))
        
            else:
                
                return tf.Variable(value, shape = (self.M,), dtype=self.dtype)
        
        else:
            
            if positive:

                assert value > 0.0

                return tf.exp(tf.Variable(np.log(value), dtype = self.dtype))

            else:

                return tf.Variable(value, dtype=self.dtype)
            
    
    def sq_exp_kernel(self, t_x1, t_x2, *args):
        
        signal_var  = args[0][0]
        lengthscale = args[0][1]
        
        t_x1 /= lengthscale
        t_x2 /= lengthscale
        
        x1x1 = tf.reduce_sum(t_x1 * t_x1, axis=1, keepdims=True)
        x2x2 = tf.reduce_sum(t_x2 * t_x2, axis=1, keepdims=True)
        
        dist = x1x1 + tf.transpose(x2x2) - 2.0  * tf.matmul(t_x1, t_x2, transpose_b=True)
    
        return signal_var * tf.exp(-0.5*dist)
    
    def periodic_kernel(self,t_X1, t_X2, *args):
    
        signal_var = args[0][0]
        gamma      = args[0][1]
        period     = args[0][2]
    
        dist_x1x2 = tf.reduce_sum(t_X1 * t_X1, axis=1, keepdims=True) + \
                    tf.transpose(tf.reduce_sum(t_X2 * t_X2, axis=1, keepdims=True)) - \
                    2*tf.matmul(t_X1, t_X2, transpose_b=True)

        sin_term  = (tf.math.sin(np.pi*(dist_x1x2**0.5)/period))**2

        return signal_var * tf.exp(- 2 * gamma * sin_term)
    
    def create_model(self, x, y, *args):
        
        if self.process_y:
            
            self.f_mu       = Regression().fit(x,y)
            self.Ymu        = self.f_mu(x)
            self.Ys2        = np.std((y - self.Ymu))
        
            y               = (y-self.Ymu)/self.Ys2
        
        self.t_X = tf.constant(x, dtype = self.dtype)
        self.t_Y = tf.constant(y, dtype = self.dtype) 
        
        self.t_N = tf.shape(self.t_Y)[0]
        self.t_D = tf.shape(self.t_Y)[1]
        self.t_Q = tf.shape(self.t_X)[0]
        self.t_M = tf.shape(self.t_X)[1]
        
        self.M = x.shape[1]
        
        if self.kernel == 'Squared Exponential':
            
            self.kernel_function = self.sq_exp_kernel

            self.signal_var  = self.init_variable(args[0][0], positive = True)
            self.lengthscale = self.init_variable([args[0][1]]*self.M, positive = True, multi = self.variable_l)
            self.noise_var   = self.init_variable(args[0][2], positive = True)
        
            self.hparamd = ['Signal Variance', 'Lengthscale']
            self.hparams = [self.signal_var, self.lengthscale]
        
        if self.kernel == 'Periodic':
            
            self.kernel_function = self.sq_exp_kernel

            self.signal_var  = self.init_variable(args[0][0], True)
            self.gamma       = self.init_variable(args[0][0], True)
            self.period      = self.init_variable(args[0][0], True)
            self.noise_var   = self.init_variable(args[0][0], True)
            
            self.p_mu        = self.init_variable(tf.log(self.t_Y), False)
            self.p_s2        = self.init_variable(1.0, True)
        
            self.hparamd = ['Signal Variance', 'Gamma', 'Period']
            self.hparams = [self.signal_var, self.gamma, self.period]
          
        self.create_kernel = lambda t_x1, t_x2: self.kernel_function(t_x1, t_x2, self.hparams)
        
        ### CREATING THE TRAINING MATRICES ###
        
        self.K_xx     = self.create_kernel(self.t_X, self.t_X) + (self.noise_var + self.jitter) * tf.eye(self.t_N, dtype = self.dtype)
        
        self.L_xx     = tf.cholesky(self.K_xx)
        
        self.logdet   = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(self.L_xx)))
        
        self.Kinv_YYt = 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.L_xx, self.t_Y, lower = True)))
        
        ### Initialising loose priors ###
        
        self.hprior = 0
        
        if self.variable_l:
            
            self.hprior += 0.5*tf.square(tf.log(self.hparams[0]))
                
            self.hprior += tf.reduce_sum(0.5*tf.square(tf.log(self.hparams[1])))
                                     
        else:
        
            for i in self.hparams:

                self.hprior += 0.5*tf.square(tf.log(i))
            
        self.noise_prior = 0.5*tf.square(tf.log(self.noise_var))
        
        ### Negative marginal log likelihood under Gaussian assumption ###
        
        if self.distribution == 'Gaussian':
        
            pi_term  = tf.constant(0.5 * np.log(2.0 * np.pi), dtype = self.dtype)
        
            self.term1 = pi_term * tf.cast(self.t_D, dtype = self.dtype) * tf.cast(self.t_N, dtype = self.dtype) \
                               + 0.5 * tf.cast(self.t_D, dtype = self.dtype) * self.logdet \
                               + self.Kinv_YYt
            
        if self.distribution == 'Poisson' and self.kernel == 'Periodic':
            
            self.Kinv = tf.cholesky_solve(self.L_xx, tf.eye(self.t_N, dtype=self.dtype))
            
            self.term1 = -tf.reduce_sum(self.t_Y*self.p_mu - tf.exp(self.p_mu + self.p_s2/2)) \
            + (1/2)*(tf.trace(self.Kinv @ (self.p_s2*tf.eye(self.t_N, dtype=self.dtype) + self.p_mu@tf.transpose(self.p_mu))) \
                     - tf.cast(self.t_N, dtype = self.dtype) + self.logdet - tf.cast(self.t_N, dtype = self.dtype)*tf.log(self.p_s2))
        
        self.objective =  self.term1 + self.hprior + self.noise_prior
    
    def optimise(self, lr, iterations, verbose):
        
            self.sess = tf.InteractiveSession()
        
            objective = self.objective ###I have only made for optimising simple likelihoods but KL divergence could also be used

            optimiser = tf.train.AdamOptimizer(learning_rate = lr).minimize(objective)
            
            self.sess.run(tf.global_variables_initializer())
            
            progress = int(np.ceil(iterations/10))
            
            for i in tqdm(range(iterations), leave = False):
                
                _, loss = self.sess.run((optimiser, objective), feed_dict = {})
                
                if verbose and (i % progress) == 0:
                        
                        print('  opt iter {:5}: objective = {}'.format(i, loss))
            
            print('Noise Variance:', self.sess.run(self.noise_var))
            
            for i in range(len(self.hparams)):
            
                print(f'{self.hparamd[i]}:', self.sess.run(self.hparams[i]))
                              
    def posterior_pred(self, x):
            
        self.Kinv_Y = tf.cholesky_solve(self.L_xx, self.t_Y)
        
        self.K_xX   = self.create_kernel(x, self.t_X)
        
        self.K_xx   = self.create_kernel(x, x)
        
        self.y_mu   = tf.matmul(self.K_xX, self.Kinv_Y)
        
        self.K_xx_d = tf.diag_part(self.K_xx) + self.noise_var * tf.ones([tf.shape(x)[0]], dtype = self.dtype)
        
        self.y_var  = self.K_xx_d - tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.L_xx, tf.transpose(self.K_xX))), axis = 0)

        self.y_var  = self.y_var[:, tf.newaxis]
        
        return self.y_mu, self.y_var
                
    def fit(self, x, y, params,
            kernel     = 'Squared Exponential', 
            lr         = 0.01, 
            iterations = 2000, 
            verbose    = True):
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Training x data
        y: ndarray
            Training y data
        params: list
            List of hyperparameters [Noise_var, Signal_var, Lengthscale - May be float or list for individual penalisation]
        kernel: str
            Choice of kernel. options ['Squared Exponential', 'Periodic'] (Default is 'Squared Exponential') Note: 'Poission' likelihood must use Periodic kernel
        lr: float
            Learning rate in training (Default is 0.01)
        iterations: int
            Number of training iterations
        verbose: Bool
            Boolean for descriptive prints
        '''
        
        if self.distribution == 'Poisson':
            
            try:
                
                assert kernel == 'Periodic'
            
            except:
                
                raise AssertionError('Must use Periodic kernel with Poisson likelihood')
        
        self.kernel = kernel
        
        self.create_model(x,y,params)
        
        self.optimise(lr, iterations, verbose)
        
        return self
    
    def predict(self, x):
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Test x data
        '''
        
        tf_mu, tf_var = self.posterior_pred(x)
        
        mu  = self.sess.run(tf_mu)
        
        var = self.sess.run(tf_var)
        
        if self.process_y:
            
            mu  = mu*self.Ys2 + self.f_mu(x)
            
            var = var * self.Ys2**2
        
        return mu, var