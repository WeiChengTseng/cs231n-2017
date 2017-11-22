from builtins import range
import numpy as np
import copy

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and 
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    M = w.shape[1]
    out = np.zeros((N, M))
    x_ori_shape = x.shape
    x = np.reshape(x, (N, -1))
    out = np.dot(x, w) + b
    x = x.reshape(x_ori_shape) 
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    D, M = w.shape
    x_ori_shape = x.shape
    db = (np.zeros((M, )))
    db = np.sum(dout, axis = 0)  
    dx = (np.dot(dout, w.T)).reshape(x_ori_shape)
    x = x.reshape((N, -1))
    dw = np.dot(x.T, dout)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    dx[x <= 0] = 0
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        x_mean = np.mean(x, axis = 0)
        x_var = np.var(x, axis = 0)
        xhat = (x - x_mean) / np.sqrt(x_var + eps)
        out = xhat * gamma + beta
        running_mean = momentum * running_mean + (1.0 - momentum) * x_mean
        running_var = momentum * running_var + (1.0 - momentum) * x_var
        cache = {}
        cache['x'] = x
        cache['xhat'] = xhat
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['x_mean'] = x_mean
        cache['x_var'] = x_var
        cache['running_mean'] = running_mean
        cache['running_var'] = running_var
        cache['eps'] = eps
        cache['N'] = N
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        cache = x
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = xhat * gamma + beta
        cache = {}
        cache['x'] = x
        cache['xhat'] = xhat
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['running_mean'] = running_mean
        cache['running_var'] = running_var
        cache['eps'] = eps   
        cache['N'] = N
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    x, xhat, gamma, beta = cache['x'], cache['xhat'], cache['gamma'] ,cache['beta']
    x_mean, x_var, eps, N = cache['x_mean'], cache['x_var'], cache['eps'], cache['N']
    
    dgamma = np.sum(xhat * dout, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # Since dx is difficult to compute, I compute it step by step.
    # Part1: compute the gradient from x_mean and x
    dxhat = dout * gamma
    dx_zero_central = dxhat / np.sqrt(x_var + eps) 
    
    # part2: compute the gradient from x_var
    dx_var = np.sum((x - x_mean) * dxhat, axis=0) * (-1. / (x_var + eps)) / (2 * ((x_var + eps) ** 0.5))
    dx_zero_central_square = np.ones((x.shape)) * dx_var / N
    
    # part3: compute the gradient from x_mean
    dx_mean = -np.sum(dx_zero_central, axis=0)
    
    # combine part1, part2 and part3
    dx = 2 * (x - x_mean) * dx_zero_central_square + dx_zero_central + np.ones((x.shape)) * dx_mean / N
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, xhat, gamma, beta = cache['x'], cache['xhat'], cache['gamma'] ,cache['beta']
    x_mean, x_var, eps, N = cache['x_mean'], cache['x_var'], cache['eps'], cache['N']
    
    dgamma = np.sum(xhat * dout, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # Since dx is difficult to compute, I compute it step by step.
    # Part1: compute the gradient from x_mean and x
    dxhat = dout * gamma
    dx_zero_central = dxhat / np.sqrt(x_var + eps) 
    
    # part2: compute the gradient from x_var
    dx_var = np.sum((x - x_mean) * dxhat, axis=0) * (-1. / (x_var + eps)) / (2 * ((x_var + eps) ** 0.5))
    dx_zero_central_square = np.ones((x.shape)) * dx_var / N
    
    # part3: compute the gradient from x_mean
    dx_mean = -np.sum(dx_zero_central, axis=0)
    
    # combine part1, part2 and part3
    dx = 2 * (x - x_mean) * dx_zero_central_square + dx_zero_central + np.ones((x.shape)) * dx_mean / N
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < p
        out = x * mask
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout
        dx[cache[1] == False] = 0
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    new_H = int(1 + (H + 2 * pad - HH) / stride)
    new_W = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, new_H, new_W))
    
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    x_pad[:, :, pad:H + pad, pad:W + pad] = x
    
    for i in range(N):
        for j in range(F):
            for k in range(new_H):
                for n in range(new_W):
                    for m in range(C):
                        out[i][j][k][n] += np.sum(x_pad[i, m, k * stride: k * stride + HH, n * stride: n * stride + WW ] * 
                                                  w[j, m, :, :])
                    out[i][j][k][n] += b[j]
                    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x = cache[0]
    w = cache[1]
    HH, WW = w.shape[2], w.shape[3]
    N, F = dout.shape[0], dout.shape[1]
    C, H, W = cache[0].shape[1], cache[0].shape[2], cache[0].shape[3]
    stride, pad = cache[3]['stride'], cache[3]['pad']
    dout_scores = np.zeros((N, F))
    dx = np.zeros(cache[0].shape)
    dw = np.zeros(cache[1].shape)
    db = np.zeros((F, ))
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    x_pad[:, :, pad:H + pad, pad:W + pad] = cache[0]
    H_pad, W_pad = x_pad[0][0].shape
    dx_pad = np.zeros(x_pad.shape)
    H_new, W_new = dout.shape[2], dout.shape[3]
    
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for n in range(W_new):
                    for m in range(F):   
                        dx_pad[i, j,  k * stride: k * stride + HH, n * stride: n * stride + WW] += dout[i][m][k][n] * w[m][j]
    dx = dx_pad[:, :, pad: pad + H, pad: pad + W]
    
    for i in range(F):
        for j in range(C):
            for k in range(HH):
                for n in range(WW):
                    for m in range(N):
                        dw[i][j][k][n] += np.sum(x_pad[m, j, k: k + H_new, n: n + W_new] * dout[m][i])
    
    for i in range(N):
        for j in range(F):
            dout_scores[i][j] = np.sum(dout[i][j])
    db = np.sum(dout_scores, axis = 0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'],  pool_param['pool_width']
    stride = pool_param['stride']
    H_out, W_out = int(1 + (H - pool_height) / stride), int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(N):
        for j in range(C):
            for k in range(H_out):
                for n in range(W_out):
                    out[i][j][k][n] = np.max(x[i, j, k * stride: k * stride + pool_height, n * stride: n * stride + pool_width])
    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    
    x = cache[0]
    N, C, H, W = x.shape
    pool_height, pool_width = cache[1]['pool_height'],  cache[1]['pool_width']
    stride = cache[1]['stride']
    H_dout, H_dout = int(1 + (H - pool_height) / stride), int(1 + (W - pool_width) / stride)
    dx = np.zeros(x.shape)
    
    for i in range(N):
        for j in range(C):
            for k in range(H_dout):
                for n in range(H_dout):
                    matrix = x[i, j, k * stride: k * stride + pool_height, n * stride: n * stride + pool_width]
                    pos_max_per_row = np.argsort(matrix)[np.arange(pool_height), pool_width - 1]
                    max_per_row = matrix[np.arange(pool_height), pos_max_per_row]
                    pos_max_col = np.argsort(max_per_row)[pool_height - 1]
                    pos_max_row = pos_max_per_row[pos_max_col]
                    dx[i][j][k * pool_height + pos_max_col][n * pool_width + pos_max_row] = dout[i][j][k][n]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    out, x, cache, gamma, beta = np.zeros((N, C, H * W)), x.reshape((N, C, H * W)), {}, gamma.reshape((C, )), beta.reshape((C, ))
    for i in range(N):
        out_T, cache[i] = batchnorm_forward(x[i].T, gamma, beta, bn_param)
        out[i] = out_T.T
    out = out.reshape((N, C, H, W))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dx, dgamma, dbeta, dout = np.zeros((N, C, W * H)), np.zeros((N, C)), np.zeros((N, C)), dout.reshape((N, C, W * H))
    for i in range(N):
        dx_T, dgamma[i], dbeta[i] = batchnorm_backward(dout[i].T, cache[i])
        dx[i] = dx_T.T
    dx, dgamma, dbeta = dx.reshape((N, C, H, W)), np.sum(dgamma, axis = 0), np.sum(dbeta, axis = 0)    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
