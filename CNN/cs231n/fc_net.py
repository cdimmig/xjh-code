from layer_utils import *

class FullyConnectedNet(object):    
    """    
    A fully-connected neural network with an arbitrary number of hidden layers,    
    ReLU nonlinearities, and a softmax loss function. This will also implement    
    dropout and batch normalization as options. For a network with L layers,    
    the architecture will be    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax    
    where batch normalization and dropout are optional, and the {...} block is    
    repeated L - 1 times.   
    Similar to the TwoLayerNet above, learnable parameters are stored in the    
    self.params dictionary and will be learned using the Solver class. 
    def __init__(self, hidden_dims, input_dim=3*32*32,  
                 num_classes=10,              
                 dropout=0, use_batchnorm=False, reg=0.0,    
                 weight_scale=1e-2, dtype=np.float32, seed=None):    
    """
    def __init__(self, hidden_dims, input_dim=3*32*32, 
                 num_classes=10,           
                 dropout=0, use_batchnorm=False, reg=0.0,      
                 weight_scale=1e-2, dtype=np.float32, seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layers_dims = [input_dim] + hidden_dims + [num_classes]
        for i in xrange(self.num_layers):    
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i+1])    
            self.params['b' + str(i+1)] = np.zeros((1, layers_dims[i+1]))    
            if self.use_batchnorm and i < len(hidden_dims): 
                self.params['gamma' + str(i+1)] = np.ones((1, layers_dims[i+1]))        
                self.params['beta' + str(i+1)] = np.zeros((1, layers_dims[i+1]))
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:    
            self.dropout_param = {'mode': 'train', 'p': dropout}    
            if seed is not None:        
                self.dropout_param['seed'] = seed
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:    
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():    
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):    
        """    
        Compute loss and gradient for the fully-connected net.    
        Input / output: Same as TwoLayerNet above.    
        """    
        X = X.astype(self.dtype)    
        mode = 'test' if y is None else 'train'    
        # Set train/test mode for batchnorm params and dropout param since they    
        # behave differently during training and testing.    
        if self.dropout_param is not None: 
            self.dropout_param['mode'] = mode    
        if self.use_batchnorm:        
        for bn_param in self.bn_params:            
            bn_param['mode'] = mode    
        scores = None    
        h, cache1, cache2, cache3, bn, out = {}, {}, {}, {}, {}, {}    
        out[0] = X

        # Forward pass: compute loss
        for i in xrange(self.num_layers-1):    
            # Unpack variables from the params dictionary    
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            if self.use_batchnorm:        
                gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]        
                h[i], cache1[i] = affine_forward(out[i], W, b)        
                bn[i], cache2[i] = batchnorm_forward(h[i], gamma, beta, self.bn_params[i])        
                out[i+1], cache3[i] = relu_forward(bn[i])    
            else:        
                out[i+1], cache3[i] = affine_relu_forward(out[i], W, b)

        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        scores, cache = affine_forward(out[self.num_layers-1], W, b)

        # If test mode return early
        if mode == 'test':   
            return scores

        loss, reg_loss, grads = 0.0, 0.0, {}
        data_loss, dscores = softmax_loss(scores, y)
        for i in xrange(self.num_layers):    
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(i+1)]*self.params['W' + str(i+1)])
        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        dout, dbn, dh = {}, {}, {}
        t = self.num_layers-1
        dout[t], grads['W'+str(t+1)], grads['b'+str(t+1)] = affine_backward(dscores, cache)
        for i in xrange(t):    
            if self.use_batchnorm:        
                dbn[t-1-i] = relu_backward(dout[t-i], cache3[t-1-i]) 
                dh[t-1-i], grads['gamma'+str(t-i)], grads['beta'+str(t-i)] = batchnorm_backward(dbn[t-1-i], cache2[t-1-i])       
                dout[t-1-i], grads['W'+str(t-i)], grads['b'+str(t-i)] = affine_backward(dh[t-1-i], cache1[t-1-i])    
            else:        
                dout[t-1-i], grads['W'+str(t-i)], grads['b'+str(t-i)] = affine_relu_backward(dout[t-i], cache3[t-1-i])

        # Add the regularization gradient contribution
        for i in xrange(self.num_layers):    
            grads['W'+str(i+1)] += self.reg * self.params['W' + str(i+1)]

        return loss, grads