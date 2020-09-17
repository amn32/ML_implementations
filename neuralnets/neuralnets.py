import warnings
import time
import tensorflow.compat.v1 as tf
import numpy                as np
import matplotlib.pyplot    as plt
import os
from IPython.display        import Image
from scipy                  import stats
from sklearn.decomposition  import PCA
from tqdm                   import tqdm_notebook as tqdm
from tensorflow             import keras

tf.disable_eager_execution()
warnings.filterwarnings('ignore')

class SLP:
    
    '''Single Layer Perceptron'''
    
    def __init__(self, x, y, iterations = 1000):
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Training x data
        y: ndraay
            Training y data
        iterations: int
            Number of iterations in training (Default is 1000)
        '''
        
        
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        self.m = self.x.shape[1]
        self.w = np.random.normal(0,1,self.m) # Randomly initialise the weights
        self.b = np.random.rand() # Randomly initialise the bias
        
        self.step       = 1
        self.error      = np.inf
        self.errors     = []
        self.iterations = iterations
        
        return
        
    def predict(self):
        
        return np.sign((self.x @ self.w.T) + self.b)
    
    def optimise(self):
        
        for i in tqdm(range(self.iterations), leave = False):

            self.error  = np.mean(self.predict() != self.y)

            self.grad   = self.predict() - self.y

            self.w_grad = self.x.T @ self.grad # Gradient wrt weights

            self.b_grad = np.sum(self.grad, axis = 0) # Gradient wrt bias

            self.w     -= self.step * self.w_grad # Take step in w

            self.b     -= self.step * self.b_grad # Take step in b

            self.errors.append(self.error)
            
            if self.error < 1e-5:
                
                break
        
        return self.w, self.b
    
    def plot_errors(self):
        
        plt.semilogy(range(len(self.errors)), self.errors)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.show()
        
        return
    
    def visualise(self, weights):
        
        plt.imshow(weights)
        plt.colorbar()
        plt.show()
        
        return

    
class MLP:
    
    '''Multilayer perceptron'''
    
    def __init__(self, output_dir = './MLP_logdir/', 
                 lr         = 0.001, 
                 nb_epochs  = 10, 
                 batch_size = 50,
                 n          = 60000,
                 img_size   = 28**2,
                 labels     = 10,
                 activation = tf.nn.relu,
                 logit_max  = tf.nn.softmax,
                 structure  = [1000,1000]):
        
        '''
        Parameters
        ----------
        
        lr: float
            Learning rate (Default is 0.001)
        nb_epochs:
            Number of training epochs (Default is 10)
        bathc_size: int
            Number of trainign exemplars run simultaneously (Default is 50)
        n: int
            Number of training exemplars
        img_size: int
            Number of pixels in image, not necessary to be square
        labels: int
            Number of class labels
        activation: tf func
            Activation function used in every layer
        logit_max:  tf func
            Activation function for the logit layer
        structure:  list
            List of layer widths
        '''
        
        self.nb_epochs  = nb_epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.nb_epochs  = nb_epochs
        self.n          = n
        self.iterations = self.n // self.batch_size
        self.output_dir = output_dir
        self.img_size   = img_size
        self.num_labels = labels
        self.activation = activation
        self.structure  = structure
        self.logit_max  = logit_max        
        self.im         = tf.placeholder(tf.float32, [None, self.img_size])
        self.labels     = tf.placeholder(tf.float32, [None, self.num_labels])
        
    def create_model(self):
        
        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):

            h = self.im
            
            for i, layer in enumerate(self.structure):
                
                h = tf.layers.dense(h, layer, 
                                    activation = self.activation, 
                                    name       = f'layer_{i+1}') 
                
                ### Add the Dense Layers

            self.logits = tf.layers.dense(h, self.num_labels, 
                                          name = 'layer_final') 
            
                ### Compute the logits
      
            self.preds  = self.logit_max(self.logits) 
        
                ### Make the predictions

    def compute_loss(self): ### Calculate the Loss function
        
        with tf.variable_scope('loss'):
            
            self.loss       = tf.losses.softmax_cross_entropy(self.labels, 
                                                              self.logits)
            
            self.loss_summ  = tf.summary.scalar("softmax_loss", self.loss)
            
    def optimizer(self): ### Optimise the Loss function
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                                     beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, 
                                                 var_list=self.model_vars)           
    def train(self, x, y):
        
        '''
        Parameters
        ----------
        
        x_train: ndarray
            Training x data
        y_train: ndarray
            Training y data
        '''
        
        self.create_model()
        self.compute_loss()
        self.optimizer()

        init    = (tf.global_variables_initializer(), 
                   tf.local_variables_initializer())

        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.InteractiveSession()
        sess.run(init)

        writer = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.Loss = []

        for epoch in tqdm(range(self.nb_epochs), leave = False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)
            x_in = x[randomize,:]
            y_in = y[randomize,:]

            for i in tqdm(range(self.iterations), leave = False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train = y_in[i*self.batch_size: (i+1)*self.batch_size]


                _ , preds, loss, loss_summ_train = sess.run([self.trainer, 
                                                             self.preds, 
                                                             self.loss, 
                                                             self.loss_summ], 
                                     feed_dict={self.im: input_x_train, 
                                                self.labels: input_y_train})

                y_preds   = np.argmax(preds, axis=1)
                y_real    = np.argmax(input_y_train, axis=1)
                acc_train = np.mean((y_preds==y_real)*1)

                writer.add_summary(loss_summ_train, epoch * self.iterations + i)
                self.Loss.append(loss/self.n)

            saver.save(sess, self.output_dir, global_step=epoch) 
            
        return sess
    
    def test(self, x): 

        batch_size_test = 20
        test_points     = x.shape[0] 
        iterations      = test_points//batch_size_test
        self.prediction = []

        for i in range(iterations):

            input_x_test = x[i*batch_size_test: (i+1)*batch_size_test]
            preds_test   = sess.run(self.preds, 
                                     feed_dict={self.im: input_x_test})

            self.prediction.append(np.argmax(preds_test, axis=1))

            if np.mod(test_points, batch_size_test) !=0:

                input_x_test = x[i*batch_size_test: -1]
                preds_test   = sess.run(self.preds, 
                                     feed_dict={self.im: input_x_test})

                self.prediction.append(np.argmax(preds, axis=1))

        return self.prediction
    
    def calc_accuracy(self, preds, y):
        
        all_preds = np.concatenate(preds, axis =0)
        y_real    = np.argmax(y, axis=1)
        acc_test  = np.mean((all_preds==y_real)*1)

        print('Test accuracy achieved: %.3f' %acc_test)

        return acc_test

    
class CNN:
    
    '''Train a convolutional neural network for image classification'''
    
    def __init__(self, output_dir = './CNN_logdir/', 
                 lr         = 0.001, 
                 nb_epochs  = 10, 
                 batch_size = 50, 
                 n          = 60000,
                 len_edge1  = 28,
                 len_edge2  = 28,
                 labels     = 10,
                 activation = tf.nn.relu,
                 logit_max  = tf.nn.softmax,
                 strides    = [1,2,2],
                 convols    = [(4,4)]*3,
                 padding    = 'SAME',
                 pooling    = True,
                 pool_size  = 2,
                 pool_strid = 2,
                 structure  = [32,64,128]):
        
        '''
        lr:         learning rate
        n:          number of training exemplars
        len_edge1:  width of image
        len_edge2:  height of image
        labels:     number of class labels
        activation: activation function used in every layer
        logit_max:  activation function for the logit layer
        structure:  list of layer widths
        strides:    list of strides, for each layer
        convols:    list of convolutions, for each layer
        padding:    specify the padding to be used with each convolution
        '''

        self.n             = n
        self.len_edge1     = len_edge1
        self.len_edge2     = len_edge2
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.output_dir    = output_dir
        self.num_labels    = labels
 
        self.im            = tf.placeholder(tf.float32, [None, self.len_edge1, self.len_edge2, 1])
        self.labels        = tf.placeholder(tf.float32, [None, self.num_labels])        
        self.structure     = structure
        self.convolutions  = convols
        self.strides       = strides
        self.logit_max     = logit_max
        self.activation    = activation
        self.padding       = padding
        
        self.pooling       = pooling
        self.pool_size     = pool_size
        self.pool_stride   = pool_strid
        
    def create_model(self):
        
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
            
            h = self.im
            
            for i, layer in enumerate(self.structure):
                
                stride = self.strides[i]
                
                convol = self.convolutions[i]
                
                h      = tf.layers.conv2d(h, layer, convol, stride, 
                                          activation = self.activation, 
                                          padding    = self.padding, 
                                          name       = f'conv_{i+1}')
                
                if self.pooling:
                    
                    h  = tf.layers.max_pooling2d(h,
                                                 self.pool_size,
                                                 self.pool_stride, 
                                                 padding = self.padding)
                
            flat  = tf.layers.flatten(h)

            self.logits = tf.layers.dense(flat, self.num_labels, name = 'layer_final')
      
            self.preds  = tf.nn.softmax(self.logits)
    
    def compute_loss(self):
        
        with tf.variable_scope('loss'):
         
            self.loss      = tf.losses.softmax_cross_entropy(self.labels, 
                                                             self.logits)
            
            self.loss_summ = tf.summary.scalar("softmax_loss", self.loss)
            
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    def train(self, x, y): 
        
        '''
        Parameters
        ----------
        
        x_train: ndarray
            Training x data
        y_train: ndarray
            Training y data
        '''
    
        self.create_model()
        self.compute_loss()
        self.optimizer()

        init    = (tf.global_variables_initializer(), 
                   tf.local_variables_initializer())

        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.InteractiveSession()
        sess.run(init)

        writer = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)

        if not os.path.exists(self.output_dir):

            os.makedirs(self.output_dir)

        self.Loss = []

        for epoch in tqdm(range(self.nb_epochs), leave = False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x.resize([self.n, self.len_edge1, self.len_edge2])
            x    = x[:, :, :,np.newaxis]
            
            x_in = x[randomize,:]
            y_in = y[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave = False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train = y_in[i*self.batch_size: (i+1)*self.batch_size]

                _ , preds, loss, loss_summ_train = sess.run([self.trainer,
                                                             self.preds, 
                                                             self.loss, 
                                                             self.loss_summ], 
                                         feed_dict={self.im: input_x_train, 
                                                    self.labels: input_y_train})

                y_preds   = np.argmax(preds, axis=1)
                y_real    = np.argmax(input_y_train, axis=1)
                acc_train = np.mean((y_preds==y_real)*1)
                writer.add_summary(loss_summ_train, epoch * self.nb_iterations + i)
                self.Loss.append(loss/self.n)
            
            saver.save(sess, self.output_dir, global_step=epoch) 

        return sess

    def test(self, x):

        batch_size_test = 20
        test_points     = x.shape[0] 
        iterations      = test_points//batch_size_test
        self.prediction = []
        
        x.resize([test_points, self.len_edge1, self.len_edge2,1])
        
        for i in range(iterations):

            input_x_test = x[i*batch_size_test: (i+1)*batch_size_test]
            preds_test   = sess.run(self.preds, feed_dict={self.im: input_x_test})
            
            self.prediction.append(np.argmax(preds_test, axis=1))

            if np.mod(test_points, batch_size_test) !=0:

                input_x_test = x[i*batch_size_test: -1]

                preds_test   = sess.run(self.preds, feed_dict={self.im: input_x_test})

                self.prediction.append(np.argmax(preds_test, axis=1))

        return self.prediction
                
    def calc_accuracy(self, preds, y):
        
        all_preds = np.concatenate(preds, axis =0)
        y_real    = np.argmax(y, axis=1)
        acc_test  = np.mean((all_preds==y_real)*1)

        print('Test accuracy achieved: %.3f' %acc_test)

        return acc_test

class NN:
    
    '''Train a neural network with dense and convolutional layers for image classification'''
    
    def __init__(self, output_dir = './NN_logdir/', 
                 lr         = 0.001, 
                 nb_epochs  = 10, 
                 batch_size = 50, 
                 n          = 60000,
                 len_edge1  = 28,
                 len_edge2  = 28,
                 labels     = 10,
                 activation = tf.nn.relu,
                 logit_max  = tf.nn.softmax,
                 strides    = [1,2,2],
                 convols    = [(4,4)]*3,
                 padding    = 'SAME',
                 pooling    = True,
                 pool_size  = 2,
                 pool_strid = 2,
                 structure1 = [32,64,128],
                 structure2 = ['conv','conv','conv']):
        
        '''
        lr:         learning rate
        n:          number of training exemplars
        len_edge1:  width of image
        len_edge2:  height of image
        labels:     number of class labels
        activation: activation function used in every layer
        logit_max:  activation function for the logit layer
        structure:  list of layer widths
        strides:    list of strides, for each layer
        convols:    list of convolutions, for each layer
        padding:    specify the padding to be used with each convolution
        '''
        
        self.n             = n
        self.len_edge1     = len_edge1
        self.len_edge2     = len_edge2
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.output_dir    = output_dir
        self.num_labels    = labels
 
        self.im            = tf.placeholder(tf.float32, [None, self.len_edge1, self.len_edge2, 1])
        self.labels        = tf.placeholder(tf.float32, [None, self.num_labels])        
        self.structure1    = structure1
        self.structure2    = structure2
        self.convolutions  = convols
        self.strides       = strides
        self.logit_max     = logit_max
        self.activation    = activation
        self.padding       = padding
        
        self.pooling       = pooling
        self.pool_size     = pool_size
        self.pool_stride   = pool_strid
        
    def create_model(self):
        
        with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
            
            h = self.im
            
            for i, layer in enumerate(self.structure2):
                
                if layer == 'conv':
                
                    depth  = self.structure1[i]
                
                    stride = self.strides[i]

                    convol = self.convolutions[i]

                    h      = tf.layers.conv2d(h, depth, convol, stride, 
                                              activation = self.activation, 
                                              padding    = self.padding, 
                                              name       = f'conv_{i+1}')

                    if self.pooling:

                        h  = tf.layers.max_pooling2d(h,
                                                     self.pool_size,
                                                     self.pool_stride, 
                                                     padding = self.padding)
                else:
                    
                        depth  = self.structure1[i]
                    
                        h = tf.layers.dense(h, depth, 
                                            activation = self.activation, 
                                            name       = f'layer_{i+1}') 
                
            flat  = tf.layers.flatten(h)

            self.logits = tf.layers.dense(flat, self.num_labels, name = 'layer_final')
      
            self.preds  = tf.nn.softmax(self.logits)
    
    def compute_loss(self):
        
        with tf.variable_scope('loss'):
         
            self.loss      = tf.losses.softmax_cross_entropy(self.labels, 
                                                             self.logits)
            
            self.loss_summ = tf.summary.scalar("softmax_loss", self.loss)
            
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    def train(self, x, y):
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Training x data
        y: ndarray
            Training y data
        '''
    
        self.create_model()
        self.compute_loss()
        self.optimizer()

        init    = (tf.global_variables_initializer(), 
                   tf.local_variables_initializer())

        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.InteractiveSession()
        sess.run(init)

        writer = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)

        if not os.path.exists(self.output_dir):

            os.makedirs(self.output_dir)

        self.Loss = []

        for epoch in tqdm(range(self.nb_epochs), leave = False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x.resize([self.n, self.len_edge1, self.len_edge2])
            x    = x[:, :, :,np.newaxis]
            
            x_in = x[randomize,:]
            y_in = y[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave = False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train = y_in[i*self.batch_size: (i+1)*self.batch_size]

                _ , preds, loss, loss_summ_train = sess.run([self.trainer,
                                                             self.preds, 
                                                             self.loss, 
                                                             self.loss_summ], 
                                         feed_dict={self.im: input_x_train, 
                                                    self.labels: input_y_train})

                y_preds   = np.argmax(preds, axis=1)
                y_real    = np.argmax(input_y_train, axis=1)
                acc_train = np.mean((y_preds==y_real)*1)
                writer.add_summary(loss_summ_train, epoch * self.nb_iterations + i)
                self.Loss.append(loss/self.n)
            
            saver.save(sess, self.output_dir, global_step=epoch) 

        return sess

    def test(self, x):

        batch_size_test = 20
        test_points     = x.shape[0] 
        iterations      = test_points//batch_size_test
        self.prediction = []
        
        x.resize([test_points, self.len_edge1, self.len_edge2,1])
        
        for i in range(iterations):

            input_x_test = x[i*batch_size_test: (i+1)*batch_size_test]
            preds_test   = sess.run(self.preds, feed_dict={self.im: input_x_test})
            
            self.prediction.append(np.argmax(preds_test, axis=1))

            if np.mod(test_points, batch_size_test) !=0:

                input_x_test = x[i*batch_size_test: -1]

                preds_test   = sess.run(self.preds, feed_dict={self.im: input_x_test})

                self.prediction.append(np.argmax(preds_test, axis=1))

        return self.prediction
                
    def calc_accuracy(self, preds, y):
        
        all_preds = np.concatenate(preds, axis =0)
        y_real    = np.argmax(y, axis=1)
        acc_test  = np.mean((all_preds==y_real)*1)

        print('Test accuracy achieved: %.3f' %acc_test)

        return acc_test
    
class MTL:
    
    def __init__(self, output_dir = './NN_logdir/',
                 lambda_    = 0.5,
                 lr         = 0.001, 
                 nb_epochs  = 10,
                 n          = 60000,
                 batch_size = 50, 
                 len_edge1  = 28,
                 len_edge2  = 28,
                 labels     = 10,
                 activation = tf.nn.relu,
                 logit_max  = tf.nn.softmax,
                 structure  = {'base':[32,64,128], 
                              'shared_dense':[3136], 
                              'task1':[1024,100,10], 
                              'task2':[1024,100,3]}):
        '''
        lambda_:    the ratio of the learning split between tasks
        lr:         learning rate
        n:          number of training exemplars
        len_edge1:  width of image
        len_edge2:  height of image
        labels:     number of class labels
        activation: activation function used in every layer
        logit_max:  activation function for the logit layer
        structure:  list of layer widths
        strides:    list of strides, for each layer
        convols:    list of convolutions, for each layer
        padding:    specify the padding to be used with each convolution
        '''  
        
        self.n               = n
        self.len_edge1       = len_edge1
        self.len_edge2       = len_edge2
        self.nb_epochs       = nb_epochs
        self.lr              = lr
        self.batch_size      = batch_size
        self.nb_epochs       = nb_epochs
        self.nb_iterations   = self.n // batch_size
        self.output_dir      = output_dir
        self.x_train         = x_train
        self.y_train_1       = y_train_1
        self.y_train_2       = y_train_2
        self.lambda_         = lambda_
        self.structure       = structure
        self.base            = self.structure['base']
        self.shared_dense    = self.structure['shared_dense']
        
        self.task1_structure = self.structure['task1'][:-1]
        self.task2_structure = self.structure['task2'][:-1]
        
        self.task1_logits    = self.structure['task1'][-1]
        self.task2_logits    = self.structure['task2'][-1]
        
        self.m               = x_train.shape[0]
        self.n_output_1      = y_train_1.shape[1]
        self.n_output_2      = y_train_2.shape[1]
        
        self.X               = tf.placeholder(tf.float32, 
                                              (None, 28, 28, 1), "X")
        
        self.y_1             = tf.placeholder(tf.float32, 
                                              (None, self.n_output_1), "y_1")
        
        self.y_2             = tf.placeholder(tf.float32, 
                                              (None, self.n_output_2), "y_2")

    
    def create_model(self):            
        
        with tf.variable_scope("MTL", reuse=tf.AUTO_REUSE):
            
            h = self.X
            
            ### Build the base layers with pooling before the shared dense layer
            
            for i, item in enumerate(self.base, 1):
                
                h = tf.layers.conv2d(h,item,(3,3), 1, 
                                     activation = tf.nn.relu, 
                                     padding = 'same', 
                                     name = f'conv_{i}')

                if i == len(self.base):
                    
                    h =  tf.layers.flatten(h)
                    
                else:
                
                    h = tf.layers.max_pooling2d(h,(2,2),2, 
                                                padding = 'same', 
                                                name = f'pool{i}')
                    
            ### Building the shared dense layer
            
            self.dense = tf.layers.dense(h, self.shared_dense[0], 
                                          activation = tf.nn.relu, 
                                          name = 'dense_layer1')
            
            ### Building Task 1 layers
            
            h1 = self.dense
            
            for i, item in enumerate(self.task1_structure, 1):
                
                h1 = tf.layers.dense(h1, item, 
                                     activation = tf.nn.relu, 
                                     name = f'dense_t1_{i}')
             
            self.logits_1 = tf.layers.dense(h1, self.task1_logits, 
                                            name = 'layer_final_1')
            
            self.pred_1   = tf.nn.softmax(self.logits_1)  
            
            ### Building Task 2 layers
            
            h2 = self.dense
            
            for i, item in enumerate(self.task2_structure, 1):
                
                h2 = tf.layers.dense(h2, item, 
                                     activation = tf.nn.relu, 
                                     name = f'dense_t2_{i}')
            
            self.logits_2 = tf.layers.dense(h2, self.task2_logits, 
                                            name = 'layer_final_2')
            
            self.pred_2   = tf.nn.softmax(self.logits_2) 
        
    def compute_loss(self):
        
        with tf.variable_scope('loss'):

            self.loss_task_1       = tf.losses.softmax_cross_entropy(self.y_1, 
                                                                     self.logits_1)
            self.loss_task_2       = tf.losses.softmax_cross_entropy(self.y_2, 
                                                                     self.logits_2) 
            
            self.loss_total        = self.lambda_*self.loss_task_1 
            
            self.loss_total       += (1-self.lambda_)*self.loss_task_2 
            
            self.loss_task_1_graph = tf.summary.scalar("softmax_loss_task_1", 
                                                       self.loss_task_1) 
            
            self.loss_task_2_graph = tf.summary.scalar("softmax_loss_task_2", 
                                                       self.loss_task_2)             
            
            self.loss_sum          = tf.summary.scalar("softmax_loss", 
                                                       self.loss_total) 
     
                
    def optimizer(self):
        
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                                     beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss_total, 
                                                 var_list=self.model_vars)
            
    def train(self, x, y1, y2):  
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Training x data
        y1: ndarray
            Training y data for task 1
        y2: ndarray
            Training y data for task 2
        '''

        self.create_model()     
        self.compute_loss()
        self.optimizer()   

        init = (tf.global_variables_initializer(),tf.local_variables_initializer())
        sess = tf.InteractiveSession()
        sess.run(init)    

        saver   = tf.train.Saver()
        summary = tf.Summary()
        writer  = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)

        if not os.path.exists(self.output_dir):

            os.makedirs(self.output_dir)

        for epoch in tqdm(range(self.nb_epochs), leave = False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x_in   = x[randomize,:]
            y_in_1 = y1[randomize,:]
            y_in_2 = y2[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave = False):

                input_x_train   = x_in[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train_1 = y_in_1[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train_2 = y_in_2[i*self.batch_size: (i+1)*self.batch_size]

                _, preds_1, preds_2, loss_1, loss_2, loss_summ = sess.run([self.trainer,
                                                                           self.pred_1, 
                                                                           self.pred_2, 
                                                                           self.loss_task_1, 
                                                                           self.loss_task_2, 
                                                                           self.loss_sum], 
                                                 feed_dict={self.X: input_x_train, 
                                                            self.y_1: input_y_train_1,
                                                            self.y_2: input_y_train_2})

                y_preds_1   = np.argmax(preds_1, axis=1)
                y_preds_2   = np.argmax(preds_2, axis=1)
                y_real_1    = np.argmax(input_y_train_1, axis=1)
                y_real_2    = np.argmax(input_y_train_2, axis=1)
                acc_train_1 = np.mean((y_preds_1==y_real_1)*1)
                acc_train_2 = np.mean((y_preds_2==y_real_2)*1)

                writer.add_summary(loss_summ, epoch * self.nb_iterations + i)

        saver.save(sess, self.output_dir, global_step=epoch) 

        return sess

    def test_MTL(self, x):

        batch_size_test = 20
        nb_test_points  = x.shape[0] 
        nb_iterations   = nb_test_points//batch_size_test
        self.preds_1         = []
        self.preds_2         = []

        for i in range(nb_iterations):

            input_x_test = x[i*batch_size_test: (i+1)*batch_size_test]

            preds_test_1, preds_test_2 = sess.run([self.pred_1, self.pred_2], 
                                     feed_dict={self.X: input_x_test})

            self.preds_1.append(np.argmax(preds_test_1, axis=1))
            self.preds_2.append(np.argmax(preds_test_2, axis=1))

            if np.mod(nb_test_points, batch_size_test) !=0:

                input_x_test               = x_test[i*batch_size_test: -1]
                preds_test_1, preds_test_2 = sess.run([self.pred_1, self.pred_2], 
                                     feed_dict={self.X: input_x_test})

                self.preds_1.append(np.argmax(preds_test_1, axis=1))
                self.preds_2.append(np.argmax(preds_test_2, axis=1))

        return self.preds_1, self.preds_2
    
    def calc_accuracy(self, preds_1, preds_2, y1, y2):

        all_preds_1 = np.concatenate(preds_1, axis =0)
        all_preds_2 = np.concatenate(preds_2, axis =0)
        y_real_1    = np.argmax(y1, axis=1)
        y_real_2    = np.argmax(y2, axis=1)
        acc_test_1  = np.mean((all_preds_1==y_real_1)*1)
        acc_test_2  = np.mean((all_preds_2==y_real_2)*1)

        print('Test accuracy achieved: %.3f' %(acc_test_1,acc_test_2))

        return acc_test_1, acc_test_2
    
class VAE:
    
    def __init__(self, output_dir = './VAE_MLP_logdir',
                 nlatent     = 2,
                 len_edge1   = 28,
                 len_edge2   = 28,
                 batch       = 64,
                 epochs      = 20,
                 alpha       = 1e-3,
                 activation  = tf.nn.relu,
                 encoder     = [512,256,128],
                 decoder     = [128,256,512],
                 activationf = tf.nn.sigmoid):
       
        dtype = 'float32'
        
        self.output_dir  = output_dir
        self.display_n   = 29
        self.digit_size  = 28
        self.z1          = stats.norm.ppf(np.linspace(0.01, 0.99, self.display_n))
        self.z2          = stats.norm.ppf(np.linspace(0.01, 0.99, self.display_n))
        self.z_grid      = np.dstack(np.meshgrid(self.z1, self.z2))
        
        self.nlatent     = nlatent
        self.batch       = batch
        self.epochs      = epochs
        self.alpha       = alpha
        self.len_edge1   = len_edge1
        self.len_edge2   = len_edge2
        self.encoder     = encoder
        self.decoder     = decoder
        self.activation  = activation
        self.activationf = activationf
        self.m           = self.len_edge1*self.len_edge2
         
        self.t_X         = tf.placeholder(dtype = dtype, shape = [None, self.m], name = 'X')

    def get_batch(self, *args, size):
        
        """ Loops through each argument in batches of [size] """

        n = len(args[0])
        if size is None or size >= n:
            yield from args
            return None
        r = np.random.permutation(n)
        for i in range(n // size + 1):
            yield (arg[r[i * size : (i + 1) * size]] for arg in args)
      
    def create_model(self):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):   
            
            h        = self.t_X
    
            for i, layer in enumerate(self.encoder):

                    h    = tf.layers.dense(h, layer, activation = self.activation, name = f'ELayer_{i+1}') 
                
        with tf.name_scope('Latent'):
    
            self.z_mean   = tf.layers.dense(h, self.nlatent, name = 'Mean')

            self.z_lvar   = tf.layers.dense(h, self.nlatent, name = 'LogVariance')
            
            self.z_var    = tf.exp(self.z_lvar, name = 'Variance')

            self.epsilon  = tf.random_normal([tf.shape(self.t_X)[0],self.nlatent])
            
            self.z_sample = self.z_mean + self.epsilon*tf.sqrt(self.z_var)
            
            self.z        = tf.placeholder_with_default(self.z_sample, shape =[None, self.nlatent], name = 'z')
            
        with tf.name_scope('Decoder'):
   
            h  = self.z

            for i, layer in enumerate(self.decoder):
            
                    h    = tf.layers.dense(h, layer, activation = self.activation, name = f'DLayer_{i}')

            self.t_X_hat  = tf.layers.dense(h, self.m, activation = self.activationf, name = 'layer_final')
            
    def compute_loss(self):
        
        with tf.name_scope('AutoEncoder'):
        
            # The normal AutoEncoder loss should measure how far our t_X_hat is from t_X
        
            loss_ae  = tf.reduce_sum(tf.square(self.t_X_hat - self.t_X), axis = 1)
        
        with tf.name_scope('KL_Divergence'):
        
            # The KL-divergence between z and a standard normal you derived earlier
            
            loss_kl  = 0.5 * tf.reduce_sum(tf.square(self.z_mean) 
                                       + self.z_var 
                                       - self.z_lvar 
                                       - tf.constant(1.), axis = 1)
        
        self.loss = tf.reduce_mean(loss_ae + loss_kl, name = 'loss')
                             
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate = self.alpha)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    
    def train(self, x): ### Train the VAE
        
        '''
        Parameters
        ----------
        
        x: ndarray
            Training x data
        '''
        
        self.create_model()
        self.compute_loss()
        self.optimizer()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        
        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.InteractiveSession()
        sess.run(init)

        writer  = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)
  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.Loss = []
        
        for i in tqdm(range(self.epochs), leave = False):
            
            self.losses  = []
            
            for xb, in self.get_batch(x_train, size = self.batch):
            
                nb = len(xb)
                sess.run(self.trainer, feed_dict = {self.t_X : xb})
                self.losses.append(nb * sess.run(self.loss, feed_dict = {self.t_X : xb}))
                self.Loss.append(self.losses[-1] / nb)
                
        return sess
    
                
    def visualise_manifold(self, x, y, sep = 2):
    
        """ Visualise the mapped 2D manifold """
    
        Z    = sess.run(self.z_mean, feed_dict = {self.t_X : x})
        feed = {self.z : self.z_grid.reshape(self.display_n * self.display_n, self.nlatent)}
        Xh   = sess.run(self.t_X_hat, feed_dict = feed).reshape(self.display_n, self.display_n, self.digit_size, self.digit_size)

        plt.figure(figsize = (12, 10))
        plt.scatter(Z[:, 0], Z[:, 1], c = y)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

        plt.figure(figsize = (12, 10))
        plt.imshow(np.block(list(map(list, Xh))), cmap = 'gray')
        start_range    = self.digit_size // 2
        end_range      = self.display_n * self.digit_size + start_range
        pixel_range    = np.arange(start_range, end_range, self.digit_size)
        sample_range_x = np.round(self.z1, 2)
        sample_range_y = np.round(self.z2, 2)
        plt.xticks(pixel_range[::sep], sample_range_x[::sep])
        plt.yticks(pixel_range[::sep], sample_range_y[::sep])
        
    def sample(self):
        
        return sess.run(self.t_X_hat, feed_dict = {self.z : np.random.normal(size = (1,self.nlatent))})
    
    def visualise_image(self, image = False):
    
        if not image:
            
            image = self.sample()
    
        plt.imshow(image.reshape(28,28))
        plt.show()
        
class DAE:
    
    '''Denoising Auto Encoder'''
    
    def __init__(self, output_dir = './DAE_logdir/', 
                 lr          = 0.001, 
                 nb_epochs   = 10, 
                 batch_size  = 50, 
                 n           = 60000,
                 len_edge1   = 28,
                 len_edge2   = 28,
                 activation1 = tf.nn.relu,
                 activation2 = tf.nn.relu,
                 activationf = tf.nn.sigmoid,
                 strides1    = [1,2,2],
                 convols1    = [(4,4)]*3,
                 padding1    = 'SAME',
                 structure1  = [32,64,128],
                 strides2    = [2,2,1],
                 convols2    = [(4,4)]*3,
                 padding2    = 'SAME',
                 structure2  = [64,32,1]):
        
        '''
        lr:         learning rate
        n:          number of training exemplars
        len_edge1:  width of image
        len_edge2:  height of image

        
        Encoder:
        
        structure1:  list of layer widths
        strides1:    list of strides, for each layer
        convols1:    list of convolutions, for each layer
        padding1:    specify the padding to be used with each convolution
        activation1: activation function used in every layer
        
        Decoder:
        
        structure2:  list of layer widths
        strides2:    list of strides, for each layer
        convols2:    list of convolutions, for each layer
        padding2:    specify the padding to be used with each convolution
        activation2: activation function used in every layer
        activationf: final activation function in decoder
        
        '''
     
        self.n             = n
        self.len_edge1     = len_edge1
        self.len_edge2     = len_edge2
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.output_dir    = output_dir
        self.noise         = tf.placeholder(tf.float32, shape=(), name="noise_factor") # Variance of noise in training
        self.im            = tf.placeholder(tf.float32, [None, 28, 28,1])
        
        #inject noise and constrain interval
        
        self.im_n          = tf.sigmoid(self.im + tf.random_normal(tf.shape(self.im),
                                                                   mean   = 0, 
                                                                   stddev = self.noise)) 
                
        self.structure1     = structure1
        self.convolutions1  = convols1
        self.strides1       = strides1
        self.activation1    = activation1
        self.padding1       = padding1
        
        self.structure2     = structure2
        self.convolutions2  = convols2
        self.strides2       = strides2
        self.activation2    = activation2
        self.activationf    = activationf
        self.padding2       = padding2
          
        self.sigmoid        = lambda x: 1/(1+np.exp(-x))
        self.inv_sigmoid    = lambda y: np.log(y/(1-y))
        self.tinv_sigmoid   = lambda z: tf.log(tf.divide(z,1-z))
        
    def create_model(self):
        
        with tf.variable_scope('Denoiser', reuse=tf.AUTO_REUSE):
            
            self.layers_list = []
            
            for i, layer1 in enumerate(self.structure1):
                
                stride = self.strides1[i]
                conv   = self.convolutions1[i]
                
                h = tf.keras.layers.Conv2D(layer1, conv, stride, 
                                     activation = self.activation1, 
                                     padding    = self.padding1, 
                                     name       = f'econv_{i+1}')
                
                self.layers_list.append(h)
            
            for j, layer2 in enumerate(self.structure2):
                
                stride = self.strides2[j]
                conv   = self.convolutions2[j]
                
                if j < len(self.structure2)-1:
                
                    k = tf.keras.layers.Conv2DTranspose(layer2, conv, stride, 
                                         activation = self.activation2, 
                                         padding    = self.padding2, 
                                         name       = f'dconv_{j+1}')
                    
                    self.layers_list.append(k)
                
                else:
                    
                    k = tf.keras.layers.Conv2D(layer2, conv, stride, 
                                         activation = self.activationf, 
                                         padding    = self.padding2, 
                                         name       = f'dconv_{j+1}')
                    
                    self.layers_list.append(k)
                
            self.model    = tf.keras.models.Sequential(self.layers_list)

            self.recon_im = self.model(self.im_n)  
            
            tf.summary.image('denoising', tf.concat([self.im, self.recon_im], axis=2))
    
    def compute_loss(self):
        
        with tf.variable_scope('loss'):
            
            self.loss      = tf.losses.mean_squared_error(self.im, self.tinv_sigmoid(self.recon_im)) 
            
            #Invert the sigmoid transformation to compute loss over [-1,1] interval
            
            self.loss_summ = tf.summary.scalar("reconstruction_loss", self.loss)
                             
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    def train(self, x, model_noise = 1): ### Train the DAE 
        
        '''
        Model_noise determines the variance of noise injected at the training stage
        '''
        
        self.create_model()
        self.compute_loss()
        self.optimizer()
        
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        
        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.InteractiveSession()
        sess.run(init)

        writer  = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)
  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.Loss     = []

        for epoch in tqdm(range(self.nb_epochs), leave=False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x    = x.reshape(-1,self.len_edge1,self.len_edge2,1)
            
            x_in = x[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave=False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]

                _ , im , im_n, recon_im, loss, loss_summ = sess.run([self.trainer,
                                                                     self.im, 
                                                                     self.im_n, 
                                                                     self.recon_im, 
                                                                     self.loss,
                                                                     self.loss_summ],
                                               feed_dict = {self.im : input_x_train, 
                                                            self.noise : model_noise})

                writer.add_summary(loss_summ, epoch * self.nb_iterations + i)
                self.Loss.append(loss)

            saver.save(sess, self.output_dir, global_step=epoch)  

        return sess
    
    def single_accuracy(self, test_image, noise , test_size = 5):

        '''Calculate the MSE of the denoised image from the original'''
        
        ims      = np.repeat(test_image,test_size)

        ims      = ims.reshape(test_size, self.len_edge1, self.len_edge2, 1)

        recon_im = sess.run(self.recon_im, feed_dict = {self.im : ims, self.noise : noise})
        
        error    = np.mean(np.square(ims - self.inv_sigmoid(np.array(recon_im))))

        return error
    
    def calc_accuracy(self, x, ims = 5, noise = 1):
        
        '''
        Calculate the MSE over the test set, 
        where the error is averaged over ims number of images, 
        with injected noise variance of noise
        '''
        x      = x.reshape(-1,self.len_edge1,self.len_edge2,1)
        
        errors = []
        
        for i in tqdm(range(x.shape[0]),leave = False):
            
            error = self.single_accuracy(x[i], noise, ims)
            
            errors.append(error)
            
        return np.mean(errors)
            
    def add_noise(self, image, noise):
        
        noisy_im = sess.run(self.im_n, feed_dict={self.im : image, self.noise : noise})
        
        return noisy_im
    
    def denoise(self, noisy_im):
        
        noisy_im = noisy_im.reshape(1,28,28,1)
        
        clean_im = sess.run(self.recon_im, feed_dict={self.im_n : noisy_im})
        
        return clean_im
    
    def visualise(self, original, noise = 1):
        
        original = original.reshape(1,28,28,1)
        
        noisy = self.add_noise(original, noise)
        clean = self.denoise(noisy)
        
        plt.figure(figsize = (12,4))
        plt.subplot(1,3,1)
        plt.imshow(original.reshape(28,28))
        plt.title('Noise-Free')
        plt.subplot(1,3,2)
        plt.imshow(noisy.reshape(28,28))
        plt.title(f'With-Noise: Noise Variance = {noise}')
        plt.subplot(1,3,3)
        plt.imshow(clean.reshape(28,28))
        plt.title('Denoised') 
        plt.show()
        
        return
    
class BNN:
    
    '''Bayesian Neural Network
    
    Parameters
    ----------
    
    lr:         learning rate
    n:          number of training exemplars
    len_edge1:  width of image
    len_edge2:  height of image
    labels:     number of class labels
    draws:      number of Monte Carlo draws over which probabilities are computed
    activation: activation function used in every layer
    logit_max:  activation function for the logit layer
    structure:  list of layer widths
    strides:    list of strides, for each layer
    convols:    list of convolutions, for each layer
    padding:    specify the padding to be used with each convolution
    '''
    
    def __init__(self, output_dir = './BNN_logdir/', 
                 lr         = 0.001, 
                 nb_epochs  = 10, 
                 batch_size = 50, 
                 n          = 60000,
                 len_edge1  = 28,
                 len_edge2  = 28,
                 labels     = 10,
                 draws      = 50,
                 activation = tf.nn.relu,
                 logit_max  = tf.nn.softmax,
                 strides    = [1,2,2],
                 convols    = [(4,4)]*3,
                 padding    = 'SAME',
                 pooling    = True,
                 pool_size  = 2,
                 pool_strid = 2,
                 structure1 = [32,64,128,None,1024,100],
                 structure2 = ['conv','conv','conv','flat','dense','dense']):
        
        self.n             = n
        self.len_edge1     = len_edge1
        self.len_edge2     = len_edge2
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.middle        = self.nb_epochs//2
        self.output_dir    = output_dir
        self.num_labels    = labels
        self.num_draws     = draws
 
        self.im            = tf.placeholder(tf.float32, [None, self.len_edge1, self.len_edge2, 1])
        self.labels        = tf.placeholder(tf.float32, [None,])        
        self.hold_prob     = tf.placeholder(tf.float32)
        
        self.structure1    = structure1
        self.structure2    = structure2
        self.convolutions  = convols
        self.strides       = strides
        self.logit_max     = logit_max
        self.activation    = activation
        self.padding       = padding
        
        self.pooling       = pooling
        self.pool_size     = pool_size
        self.pool_stride   = pool_strid
        
        self.kld           = lambda q,p : tfp.distributions.kl_divergence(q,p)
        
    def create_model(self):
        
        with tf.variable_scope('BNN', reuse=tf.AUTO_REUSE):
            
            h = self.im
            
            self.layers_list = []
            
            for i, layer in enumerate(self.structure2):
                
                if layer == 'conv':
                
                    depth  = self.structure1[i]
                
                    stride = self.strides[i]

                    convol = self.convolutions[i]

                    h      = tfp.layers.Convolution2DReparameterization(filters = depth, 
                                                             kernel_size = convol, 
                                                             strides = stride, 
                                                             activation = self.activation, 
                                                             padding    = self.padding, 
                                                             name       = f'conv_{i+1}_VALID')
                    
                    self.layers_list.append(h)

                    if self.pooling:

                        h  = tf.keras.layers.MaxPooling2D(self.pool_size,
                                                           self.pool_stride, 
                                                           padding = self.padding)
                        
                        self.layers_list.append(h)
                        
                elif layer == 'flat':
                    
                    h  = tf.keras.layers.Flatten()
                    
                    self.layers_list.append(h)
                
                elif layer == 'dense':
                    
                    depth  = self.structure1[i]

                    h = tfp.layers.DenseFlipout(depth,
                                                activation = self.activation, 
                                                name       = f'layer_{i+1}_VALID') 
                    
                    self.layers_list.append(h)
                
            self.dropout = tf.keras.layers.Dropout(self.hold_prob)
            
            self.layers_list.append(self.dropout)
                
            self.final = tfp.layers.DenseFlipout(self.num_labels,
                                                  name = 'layer_final_VALID')
            
            self.layers_list.append(self.final)
            
            self.model         = tf.keras.models.Sequential(self.layers_list)   
            
            self.valid_layers  = [layer for layer in self.model.layers if 'VALID' in layer.name]
        
            self.logits        = self.model(self.im)
            
            self.preds         = tf.argmax(self.logits, axis = 1)
     
    def compute_loss(self): ### Calculate the Loss function
        
        with tf.variable_scope('loss'):
            
            self.labels_dist   = tfp.distributions.Categorical(logits = self.logits)
            self.nll           = -tf.reduce_mean(self.labels_dist.log_prob(self.labels))
            self.kl            = sum(self.model.losses) / self.n
            self.ELBO          = self.nll + self.kl
            
            self.accuracy, self.accuracy_update = tf.metrics.accuracy(labels=self.labels, predictions=self.preds)
            
            self.loss_summ  = tf.summary.scalar("softmax_loss", self.ELBO)
            
    def optimizer(self): ### Optimise the Loss function
        
        with tf.variable_scope('optimizer'):
            
            optimiser          = tf.train.AdamOptimizer(learning_rate=self.lr)
            
            self.model_vars    = tf.trainable_variables()
            
            self.trainer       = optimiser.minimize(self.ELBO, var_list=self.model_vars)  
    
    def save_posterior(self):
      
        layer_name = [layer.name for layer in self.valid_layers]
    
        means      = [self.sess.run(layer.kernel_posterior.mean()) for layer in self.valid_layers]

        stds       = [self.sess.run(layer.kernel_posterior.stddev()) for layer in self.valid_layers]

        return (layer_name, means, stds)
    
    def train(self, x, y): 
    
        self.create_model()
        self.compute_loss()
        self.optimizer()
        
        init    = tf.group(tf.global_variables_initializer(), 
                   tf.local_variables_initializer())

        saver   = tf.train.Saver()
        summary = tf.Summary()
        sess    = tf.Session()
        sess.run(init)

        self.sess = sess
        
        writer = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(sess.graph)

        if not os.path.exists(self.output_dir):

            os.makedirs(self.output_dir)

        for epoch in tqdm(range(self.nb_epochs), leave = False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x.resize([self.n, self.len_edge1, self.len_edge2, 1])
            
            x_in = x[randomize,:]
            y_in = y[randomize]
            
            for i in tqdm(range(self.nb_iterations), leave = False):
                
                if i == 0 and epoch ==0: self.posterior1 = self.save_posterior()

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]
                input_y_train = y_in[i*self.batch_size: (i+1)*self.batch_size]

                _  = sess.run([self.trainer, self.accuracy_update], 
                     feed_dict={self.im: input_x_train, 
                                self.labels: input_y_train,
                                self.hold_prob:0.5})

                y_preds   = np.argmax(preds, axis=0)
                y_real    = np.argmax(input_y_train, axis=0)
                acc_train = np.mean((y_preds==y_real)*1)
                    
            if epoch == self.middle: self.posterior2 = self.save_posterior()

            if epoch == self.nb_epochs-1: self.posterior3 = self.save_posterior()
            
            saver.save(sess, self.output_dir, global_step=epoch) 
            
        return self.sess

    def test(self, x):

        batch_size_test = 20
        test_points     = x.shape[0] 
        iterations      = test_points//batch_size_test
        self.prediction = []
        
        x.resize([test_points, self.len_edge1, self.len_edge2,1])
        
        for i in range(iterations):

            input_x_test = x[i*batch_size_test: (i+1)*batch_size_test]
            preds_test   = self.sess.run(self.preds, feed_dict={self.im: input_x_test, self.hold_prob:0.5})
            
            self.prediction.append(preds_test)

            if np.mod(test_points, batch_size_test) !=0:

                input_x_test = x[i*batch_size_test: -1]

                preds_test   = self.sess.run(self.preds, feed_dict={self.im: input_x_test})

                self.prediction.append(preds_test)

        return self.prediction
                
    def calc_accuracy(self, preds, y):
        
        all_preds = np.concatenate(preds)
        acc_test  = np.mean((all_preds==y)*1)

        print('Test accuracy achieved: %.3f' %acc_test)

        return acc_test
    
    def sample_predictions(self, test_im, samples):
        
        original = test_im.copy().reshape(self.len_edge1,self.len_edge2)   
        
        preds    = []
        
        test_im.resize([1, self.len_edge1, self.len_edge2,1])
        
        for i in range(samples):
        
            pred     = self.sess.run(self.preds, feed_dict={self.im: test_im, self.hold_prob:0.5})
            
            preds.append(pred)
            
        class_counts = np.array([list(preds).count(i) for i in range(self.num_labels)])
        
        class_probs  = class_counts/samples
        
        plt.figure(figsize = (8,4))
        plt.suptitle('Monte Carlo Probabilities for test image')
        plt.subplot(1,2,1)
        plt.imshow(original)
        plt.title('Test Image')
        plt.subplot(1,2,2)
        sns.barplot(np.arange(10), class_probs, alpha = 1)
        plt.title('Prob of Class')
        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        plt.show()
            
        return preds
    
    def plot_posterior(self):
        
        name1, mean1, std1 = self.posterior1
        name2, mean2, std2 = self.posterior2
        name3, mean3, std3 = self.posterior3
        
        ### Means ###
        
        plt.figure(figsize = (12,4))
        plt.suptitle('Plot of posterior weight means over iterations')
        
        plt.subplot(1,3,1)
        plt.title('Iteration 1')
        
        for i in range(len(mean1)):
            sns.distplot(mean1[i].reshape(-1,1))
        plt.legend(name1)

        plt.subplot(1,3,2)
        plt.title(f'Iteration {self.middle}')
        
        for i in range(len(mean2)):
            sns.distplot(mean2[i].reshape(-1,1))


        plt.subplot(1,3,3)
        plt.title(f'Iteration {self.nb_epochs}')
        
        for i in range(len(mean3)):
            sns.distplot(mean3[i].reshape(-1,1))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        plt.show()
        
        ### STDS ###
        
        plt.figure(figsize = (12,4))
        plt.suptitle('Plot of posterior weight stds over iterations')
        
        plt.subplot(1,3,1)
        plt.title('Iteration 1')
        
        for i in range(len(std1)):
            sns.distplot(std1[i].reshape(-1,1), label = name1[i])

        plt.subplot(1,3,2)
        plt.title(f'Iteration {self.middle}')
        
        for i in range(len(std2)):
            sns.distplot(std2[i].reshape(-1,1), label = name2[i])


        plt.subplot(1,3,3)
        plt.title(f'Iteration {self.nb_epochs}')
        
        for i in range(len(std3)):
            sns.distplot(std3[i].reshape(-1,1), label = name3[i])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        plt.show()

class TFSOM:
    
    def __init__(self, xdim, ydim, n_nodes, lr = 0.5, r = 0.5, iterations = 50):
        
        tf.reset_default_graph()
        self.xdim      = xdim
        self.ydim      = ydim
        self.lr        = lr
        self.r         = r
        self.n_nodes   = n_nodes
        self.iters     = iterations
        self.dtype     = 'float32'
        self.size      = xdim*ydim
        self.t_size    = tf.constant(self.size, 'int32')
        self.progress  = tf.placeholder(self.dtype)
        self.nodes     = np.array([[i,j] for i in range(xdim) for j in range(ydim)])
        self.div       = tf.divide(self.progress, self.iters)
        self.map       = tf.Graph()
        self.euclid    = lambda p,q: tf.reduce_sum(tf.square(tf.subtract(p,q)), axis = 1)
        self.weights   = tf.Variable(tf.random_normal([self.t_size, self.n_nodes]), dtype = self.dtype, name = 'weights')
        self.x         = tf.placeholder(dtype = self.dtype, shape = (self.n_nodes, 1), name = 'data')
        
        self.big_x     = tf.transpose(tf.broadcast_to(self.x, (self.n_nodes, self.t_size)))
        self.dist      = tf.sqrt(self.euclid(self.weights,self.big_x))
        self.best      = tf.argmin(self.dist, axis = 0)
        self.best_loc  = tf.reshape(tf.slice(self.nodes, tf.pad(tf.reshape(self.best, [1]), np.array([[0, 1]])), tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)), [2])
        self.dlr       = tf.multiply(self.lr, 1 - self.div)
        self.radius    = tf.multiply(self.r,  1 - self.div)
        
    def optimiser(self):
        
        self.b_matrix  = tf.broadcast_to(self.best_loc, (self.size,2))
        self.b_dist    = self.euclid(self.nodes, self.b_matrix)
        self.surround  = tf.exp(tf.negative(tf.divide(tf.cast(self.b_dist, self.dtype), tf.cast(tf.square(self.radius), self.dtype))))
        self.lr_matrix = tf.multiply(self.dlr, self.surround)    
        self.factor    = tf.stack([tf.broadcast_to(tf.slice(self.lr_matrix, np.array([node]), np.array([1])), (self.n_nodes,)) for node in range(self.size)])        
        self.factor   *= tf.subtract(tf.squeeze(tf.stack([self.x for i in range(self.size)]),2), self.weights)                              
        fitted_weights = tf.add(self.weights, self.factor)
        
        self.optimise  = tf.assign(self.weights, fitted_weights)                                       
        
    def train(self, x_train):
        
        self.sess = tf.Session()
        init      = tf.global_variables_initializer()
        
        self.sess.run(init)
        
        self.optimiser()
        
        for iteration in tqdm(range(self.iters), leave = False):
            
            for train in x_train:
                
                self.sess.run(self.optimise, feed_dict={self.x: train.reshape(-1,1),self.progress: iteration})

        self.fitted_weights  = np.array(self.sess.run(self.weights))

    def fit(self, x):
        
        fitted         = np.empty((x.shape[0],2))

        node_distances = np.empty((x.shape[0], self.nodes.shape[0]))
        
        for j in range(x.shape[0]):
            
            distances_to_nodes = [np.linalg.norm(x[j,:] - self.fitted_weights[i]) for i in range(self.fitted_weights.shape[0])]
            
            node_distances[j]  = distances_to_nodes
            
            best_node          = self.nodes[np.argmin(distances_to_nodes)]
            
            fitted[j]          = best_node
        
        return fitted, node_distances
                                   
class DimAE:
    
    '''Dimensionality Reducing Auto Encoder'''
    
    def __init__(self, output_dir = './dimAE_logdir/', 
                 lr          = 0.001, 
                 nb_epochs   = 1, 
                 batch_size  = 50, 
                 n           = 60000,
                 num_chans   = 100,
                 activation  = tf.nn.relu,
                 encoder     = [80,40],
                 latent_dim  = 10,
                 verbose     = False):
        
     
        self.n             = n
        self.num_chans     = num_chans
        self.nb_epochs     = nb_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.nb_epochs     = nb_epochs
        self.nb_iterations = self.n // batch_size
        self.output_dir    = output_dir
        self.input         = tf.placeholder(tf.float32, [None, self.num_chans])
        self.dtype         = 'float32'

        self.activation    = activation
        self.encoder       = encoder
        self.latent_dim    = latent_dim
        self.verbose       = verbose
        
        
    def create_model(self):
        
        tf.keras.backend.set_floatx('float64')
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            
            self.encoder_layers = []
            
            for i, layer1 in enumerate(self.encoder):
                
                h = tf.keras.layers.Dense(layer1, dtype = self.dtype,
                                     activation = self.activation,
                                     name       = f'encoder_{i+1}')
                
                self.encoder_layers.append(h)
            
            self.latent_layer =  tf.keras.layers.Dense(self.latent_dim,
                                                      dtype = self.dtype,
                                                      activation = self.activation,
                                                      name       = f'latent')
            
            self.encoder_layers.append(self.latent_layer)
            
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE): 

            self.decoder_layers = []
            
            for j, layer2 in enumerate(self.encoder[::-1]):
                      
                k = tf.keras.layers.Dense(layer2,  dtype = self.dtype,
                                     activation = self.activation,  
                                     name       = f'decoder{j+1}')

                self.decoder_layers.append(k)
                
            k =  tf.keras.layers.Dense(self.num_chans,
                                       dtype = self.dtype,
                                       activation = self.activation,
                                       name       = f'last') 
            
            self.decoder_layers.append(k)
            
        self.encoder = tf.keras.models.Sequential(self.encoder_layers)
        
        self.decoder = tf.keras.models.Sequential(self.decoder_layers)
                
        self.model   = tf.keras.models.Sequential([self.encoder,self.decoder])

        self.recon = self.model(self.input)  
        
        if self.verbose:
            
            print(dimae.model.summary())
            print(dimae.encoder.summary()) 
            print(dimae.decoder.summary())
    
    def compute_loss(self):
        
        with tf.variable_scope('loss'):
            
            self.loss      = tf.losses.mean_squared_error(self.input, self.recon) 
            
            self.loss_summ = tf.summary.scalar("reconstruction_loss", self.loss)
                             
    def optimizer(self):
        
        with tf.variable_scope('optimizer'):
            
            optimizer       = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            
            self.model_vars = tf.trainable_variables()
            
            self.trainer    = optimizer.minimize(self.loss, var_list=self.model_vars)
            
    def train(self, x): ### Train 

        self.create_model()
        self.compute_loss()
        self.optimizer()
        
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        
        saver        = tf.train.Saver()
        summary      = tf.Summary()
        self.sess    = tf.InteractiveSession()
        self.sess.run(init)

        writer  = tf.summary.FileWriter(self.output_dir)
        writer.add_graph(self.sess.graph)
  
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.Loss     = []

        for epoch in tqdm(range(self.nb_epochs), leave=False):

            randomize = np.arange(x.shape[0])
            np.random.shuffle(randomize)

            x    = x.reshape(-1,self.num_chans)
            
            x_in = x[randomize,:]

            for i in tqdm(range(self.nb_iterations), leave=False):

                input_x_train = x_in[i*self.batch_size: (i+1)*self.batch_size]

                _ , x , recon, loss, loss_summ = self.sess.run([self.trainer,
                                                           self.input,  
                                                           self.recon, 
                                                           self.loss,
                                                           self.loss_summ],
                                       feed_dict = {self.input : input_x_train})
                
                if self.verbose:
                    
                    if i % (self.nb_iterations/5) == 0:
                    
                        print(loss)
                
                writer.add_summary(loss_summ, epoch * self.nb_iterations + i)
                self.Loss.append(loss)

            saver.save(self.sess, self.output_dir, global_step=epoch)  

        return
    
    def reduce(self, x):
        
        latent = self.sess.run(self.encoder(x))

        return latent

        
