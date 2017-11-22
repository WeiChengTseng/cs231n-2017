import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D = W.shape[0]
    num_class = W.shape[1]
    num_image = X.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    score = np.zeros((num_image, num_class))
    score_exp = np.zeros((num_image, num_class))
    score_exp_sum = np.zeros((num_image, ))
    for i in xrange(num_image):
        score[i] = X[i].dot(W)
        score[i] -= np.max(score[i])
        score_exp[i] = np.exp(score[i])
        score_exp_sum[i] = np.sum(score_exp[i])
        loss -= np.log(score_exp[i][y[i]] / score_exp_sum[i])
    loss /= num_image
    loss += reg * np.sum(W * W)
    
    for i in xrange(D):
        for j in xrange(num_class):
            for k in xrange(num_image):
                if j == y[k]:
                    dW[i][j] += -X[k][i] + X[k][i] * score_exp[k][j] / score_exp_sum[k]
                else:
                    dW[i][j] += X[k][i] * score_exp[k][j] / score_exp_sum[k]
    
    dW = dW / num_image
    dW += reg * W * 2
    
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D = W.shape[0]
    num_class = W.shape[1]
    num_image = X.shape[0]
    score = np.zeros((num_image, num_class))
    
    score = np.dot(X, W)
    score = score - (np.max(score, axis = 1)).reshape((num_image, 1))
    score_exp = np.exp(score)
    score_exp_sum = np.sum(score_exp, axis = 1)
    score_exp = score_exp / score_exp_sum.reshape((num_image, 1))
    loss = (np.sum(-np.log(score_exp[np.arange(num_image), y]))) / num_image
    loss += reg * np.sum(W * W)
    
    score_exp[np.arange(num_image), y] -= 1
    dW = (X.T).dot(score_exp)
    dW = dW / num_image + reg * W *2
    
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

