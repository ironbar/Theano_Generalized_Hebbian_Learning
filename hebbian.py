"""
This tutorial introduces generalized hebbian learning with MNIST
Is based on the code logistic_sgd.py which can be found on the following link
http://deeplearning.net/tutorial/logreg.html#logreg

Theory for generalized hebbian learning can be found on
http://courses.cs.washington.edu/courses/cse528/09sp/sanger_pca_nn.pdf

@author: guillermobarbadillo
"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import time
import numpy
import theano
import theano.tensor as T
try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images #http://deeplearning.net/tutorial/code/utils.py
import matplotlib.pyplot as plt

class HebbianLayer(object):
    """
    Hebbian Layer Class
    The hebbian is fully described by a weight matrix :math:`W`
    Learning is made using the generalized hebbian learning rule
    """

    def __init__(self, rng, input, n_in, n_out, W=None,
                 activation=T.nnet.sigmoid,batch_size=10):
        """ 
        Initialize the parameters of the hebbian layer
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units
        
        :type rng: numpy.random.RandomState
        :param rng: random number generator
        
        :type activation: theano function
        :param activation: the activation function used in the layer
        
        :type batch_size: int
        :param batch_size: size of batch used for training
        """
        self.input=input
        self.n_in=n_in
        self.n_out=n_out
        self.batch_size=batch_size
        self.rng=rng
        self.activation=activation
        #Use xabier initialization for the weights
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)   #weights of the layer
        self.W_temp=theano.shared(value=W_values, name='W_temp', borrow=False)  #temporal storare for the weights
        self.W_variation=theano.shared(value=W_values, name='W_variation', borrow=False)    #Stores the variation of the weights
        self.W_acum = theano.shared(value=numpy.zeros(shape=(n_in, n_out),dtype=theano.config.floatX), name='W_acum', borrow=True)  #Stores the change in weights for adagrad
        self.W_block = theano.shared(value=numpy.ones(shape=(n_in, n_out),dtype=theano.config.floatX), name='W_block', borrow=True)  #Used for blocking the learning of units
        self.blocked_units = numpy.zeros(n_out) #binary variable that stores the states of the weights (1 means blocked, 0 unblocked)
        
        #Lower triangular matrix, necessary for the hebbian learning rule
        LT=numpy.ones((self.n_out,self.n_out,self.n_in),dtype=theano.config.floatX)
        for i in range(self.n_out-1):
            LT[i,i+1:]=0
        self.LT = theano.shared(value=LT, name='LT', borrow=True)   
        
        lin_output = T.dot(input, self.W)   #Not using bias by the moment
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W]
        
    def weightVariation(self):
        """
        Returns the variation in weight
        W size=(n_in, n_out)
        """
        variation=T.mean(abs(self.W-self.W_temp),axis=0)/T.mean(abs(self.W),axis=0)
        return variation
            
    def updateTempWeight(self):
        """
        Returns the update to the temporal variable of the weights
        """
        return [(self.W_temp,self.W)]
    
    def hebbianRule(self):
        """
        Returns the generalized hebbian rule that will be used for updating the weights
        """
        ####NewNew formulation
        hebbian_rule=T.repeat(T.reshape(self.input,(self.batch_size,self.n_in,1)),self.n_out,axis=2)  #batch_size, n_in, n_out
        hebbian_rule=hebbian_rule*T.repeat(T.reshape(self.output,(self.batch_size,1,self.n_out)),self.n_in,axis=1) #batch_size, n_in, n_out
        
        gen_term=T.repeat(T.reshape(T.transpose(self.W),(1,self.n_out,self.n_in)),self.n_out,axis=0)
        gen_term=gen_term*self.LT
        gen_term=T.dot(self.output,gen_term)
        gen_term=T.transpose(gen_term,(0,2,1))
        gen_term=T.repeat(T.reshape(self.output,(self.batch_size,1,self.n_out)),self.n_in,axis=1)*gen_term
        
        hebbian_rule=hebbian_rule-gen_term  #batch_size, n_in, n_out
        hebbian_rule=T.mean(hebbian_rule,axis=0)     # n_in, n_out
        return hebbian_rule
    def updates(self,lr,method='cte'):
        """
        Returns the updates to the weights given a learning rate
        T.arange
        self.b.dimshuffle('x', 0, 'x', 'x')
        """
        if method=='cte':
            return [(self.W,self.W+lr*self.W_block*self.hebbianRule()),
                    (self.W_variation,self.hebbianRule())]
        elif method=='adagrad2':    #Not recommended
            rho = 0.1
            decay=0.9
            updates = []
            updates.append((self.W_variation,self.hebbianRule()))
            updates.append((self.W_acum,self.W_acum*decay+(1-decay)*self.hebbianRule()**2))
            updates.append((self.W,self.W+self.W_block*lr*self.hebbianRule()*rho/(rho+(self.W_acum*decay+(1-decay)*self.hebbianRule()**2)**0.5)))
            return updates
        else:
            raise("Error in selection update method")
            
    def blockUnitLearning(self,i):
        """
        Stops the learning of the selected unit
        """
        if self.blocked_units[i]==False:
            if i==0 or numpy.min(self.blocked_units[0:i])==1:   #We can only block one unit if the previous ones are blocked
                self.blocked_units[i]=True
                W_block = self.W_block.get_value(borrow=True)
                W_block[:,i]=0
                self.W_block.set_value(W_block,borrow=True)              
                
    def resetWeights(self):
        """
        Resets the weights of the layer to aleatory values
        """
        #Use xabier initialization for the weights
        W_values = numpy.asarray(
            self.rng.uniform(
                low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)
            ),
            dtype=theano.config.floatX
        )
        if self.activation == T.nnet.sigmoid:
            W_values *= 4
        self.W.set_value(W_values,borrow=True)
        self.blocked_units = numpy.zeros(self.n_out)
        self.W_block.set_value(numpy.ones(shape=(self.n_in, self.n_out),dtype=theano.config.floatX),borrow=True)

class HebbianNet(object):
    """
    Class that builds a stack of hebbian layers
    """
    def __init__(self, rng, input, n_in, n_hidden_list,
                 activation=T.nnet.sigmoid,batch_size=10):
        """ 
        Initialize the parameters of the hebbian layer
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units

        :type n_hidden_list: list of int
        :param n_hidden_list: number of hidden units in each layer
        
        :type rng: numpy.random.RandomState
        :param rng: random number generator
        
        :type activation: theano function
        :param activation: the activation function used in the layer
        
        :type batch_size: int
        :param batch_size: size of batch used for training
        """
        #Create the layers and link them
        self.layers=[]
        for n_hidden in n_hidden_list:
            if self.layers==[]:
                layer=HebbianLayer(input=input,rng=rng, n_in=n_in, n_out=n_hidden,
                              batch_size=batch_size,activation=activation)
            else:
                layer=HebbianLayer(input=self.layers[-1].output,rng=rng, n_in=self.layers[-1].n_out, n_out=n_hidden,
                              batch_size=batch_size,activation=activation)  
            self.layers.append(layer)
        self.output=self.layers[-1].output
        
    def updates(self,lr,method='cte'):
        """
        Returns the updates to the weights given a learning rate
        and an optimization method
        """
        updates=[]
        for layer in self.layers:
            updates=updates+layer.updates(lr=lr,method=method)
        return updates
        
    def weightVariation(self):
        """
        Returns the variation in weight of the layers
        """
        variations=[]
        for layer in self.layers:
            variations.append(layer.weightVariation())
        return variations
        
    def updateTempWeight(self):
        """
        Returns the update to the temporal variable of the weights
        """
        updates=[]
        for layer in self.layers:
            updates=updates+layer.updateTempWeight()
        return updates                

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    print 'Size of train,valid and test set:',len(train_set[0]),len(valid_set[0]),len(test_set[0])
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        data_x=numpy.round(data_x,0)
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
#                                               dtype='int8'),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
#                                               dtype='int32'),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y   #I've changed this line because I couldn't access the values
#        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    print '... loaded train data'
    test_set_x, test_set_y = shared_dataset(test_set)
    print '... loaded test data'
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    print '... loaded valid data'

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=5e-3, n_epochs=50,
                           dataset='mnist.pkl.gz',
                           batch_size=50,n_hidden_list=[16],activation=T.nnet.sigmoid,
                           block_learning=True,n_tsne=2000,endingThreshold=0.001):
    """
    Demonstrate stochastic gradient descent optimization of a hebbian net

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
                 
    :type batch_size: int
    :param batch_size: size of the batch for training
    
    :type n_hidden_list: list of int
    :param n_hidden_list: number of hidden units in each layer
    
    :type activation: theano function
    :param activation: the activation function used in the layer
    
    :type block_learning: bool
    :param block_learning: if true the weights are blocked once their variation is small
    
    :type n_tsne: int
    :param n_tsne: number of instances saved for later plotting with tsne 
    https://lvdmaaten.github.io/tsne/ 
    
    :type endingThreshold: float
    :param endingThreshold: minimun weight variation for considered the end of training

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    rate = T.fscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    rng = numpy.random.RandomState(1234)
    classifier = HebbianNet(rng=rng,input=x, n_in=28 * 28, n_hidden_list=n_hidden_list,
                              batch_size=batch_size,activation=activation)
                              
    #Compile a theano function that returns the output of the net and updates the weights
    train_model = theano.function(
        inputs=[index,rate],
        outputs=[classifier.output],
        updates=classifier.updates(rate,method='cte'),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    #Compile a theano function that returns the variation in weights and updates the temporal weights
    getVariation = theano.function(
        inputs=[],
        outputs=classifier.weightVariation(),
        updates=classifier.updateTempWeight(),
    )

    #Define useful functions for training
    def saveWeightImage():
        """
        Saves an image of the weigths of each layer.
        """
        for i in range(len(classifier.layers)):
            layer=classifier.layers[i]
            image = Image.fromarray(
                tile_raster_images(
                    X=layer.W.get_value(borrow=True).T,
                    img_shape=(int(numpy.sqrt(layer.n_in)), int(numpy.sqrt(layer.n_in))),
                    tile_shape=(int(numpy.sqrt(layer.n_out)), int(numpy.sqrt(layer.n_out))),
                    tile_spacing=(1, 1)
                )
            )
            image_name='hebbian_plots/lr'+str(learning_rate)+'_layer'+str(i)+'_nHidden'+str(layer.n_out)+'_epoch'+str(epoch)+'_bs'+str(batch_size)+'.png'
            image.save(image_name)
      
    weight_var=[]   #Storage for the weight variation in each epoch
    for j in range(len(n_hidden_list)):
        weight_var.append([])
        
    def printTrainingState():
        """
        Prints the evolution of the training in the console
        """
        print 'Epoch:\t',epoch,
        weight_var_temp=getVariation()
        #Add the variations to the list
        for j in range(len(n_hidden_list)):
            weight_var[j].append(weight_var_temp[j])
        for j in range(len(n_hidden_list)):
            print 'Layer',j,
            n=5
            if block_learning:
                print 'n_blocked:',numpy.sum(classifier.layers[j].blocked_units),
            print 'weight_var',
            step=int(len(weight_var_temp[j])/(n-1))
            for i in range(n):
                if i*step>=len(weight_var_temp[j]):
                    print weight_var_temp[j][-1],
                else:
                    print numpy.mean(weight_var_temp[j][i*step:(i+1)*step]),
        print
        
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    print 'Learning rate: ',learning_rate
    print 'n_epochs: ',n_epochs
    print 'N hidden_list:',n_hidden_list
    print 'batch_size:',batch_size
    print 'Activation:',activation
    print 'block_learning:',block_learning
    print 'endingThreshold:',endingThreshold 
    print 

    done_looping = False
    epoch = 0
    t=time.time()
    try:
        os.stat('hebbian_plots/')
    except:
        os.mkdir('hebbian_plots/') 
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index,float(learning_rate))
            
        saveWeightImage()
        printTrainingState()
        #Check for ending and blocking blocking condition
        done_looping=True
        for j in range(len(n_hidden_list)):
            for i in range(len(weight_var[j][-1])):
                if weight_var[j][-1][i]<endingThreshold:
                    if block_learning:
                        classifier.layers[j].blockUnitLearning(i)
                else:
                    done_looping=False
                    break

    #Train has ended            
    if done_looping:
        print 'Training finished'
    print 'The training took',(time.time()-t)/60,'minutes'
    #Plot the variation of the weights over the training epochs     
    for j in range(len(n_hidden_list)):
        weight_var_temp = numpy.asarray(weight_var[int(j)],dtype=numpy.float32)
        plt.figure('Weight variation layer '+str(j))
        plt.subplot(111)
        plt.clf()
        for i in range(weight_var_temp.shape[1]):
            plt.plot(range(weight_var_temp.shape[0]),weight_var_temp[:,i],label=str(i))
        plt.legend()
        plt.subplot(111).set_xscale('log')
        plt.subplot(111).set_yscale('log')
        plt.savefig('hebbian_plots/Weight variation layer '+str(j)+'.png')  
    plt.show() 
    #Save data for visualizing with tsne
    if n_tsne>0:
        print 'End of training, save some results for visualizing with t-sne'
        output=[]
        for i in range(int(n_tsne/batch_size)):
            output.append(train_model(i,0)[0])
        output=numpy.array(output,dtype=numpy.float32)
        print 'output.shape',output.shape
        n_tsne=output.shape[0]*output.shape[1]
        output=numpy.reshape(output,(n_tsne,output.shape[2]))
        print 'output.shape',output.shape
        labels = train_set_y.get_value(borrow=True)
        print 'labels.shape',labels.shape
        print 'inputs.shape',labels.shape
        numpy.savetxt('hebbian_plots/output_lr'+str(learning_rate)+'_nHidden'+str(n_hidden_list)+'_epoch'+str(epoch)+'_bs'+str(batch_size)+'.txt',output,fmt='%f')
        numpy.savetxt('hebbian_plots/labels_lr'+str(learning_rate)+'_nHidden'+str(n_hidden_list)+'_epoch'+str(epoch)+'_bs'+str(batch_size)+'.txt',labels[0:n_tsne],fmt='%f')
            

if __name__ == '__main__':
    n_epochs=200
    n_hidden_list=[9]
    sgd_optimization_mnist(n_epochs=n_epochs,n_hidden_list=n_hidden_list)
