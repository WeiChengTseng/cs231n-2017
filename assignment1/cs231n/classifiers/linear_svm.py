import numpy as np
from random import shuffle
from past.builtins import xrange
  
def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    D = W.shape[0]    
    margin_shifted = 0
    margin = np.zeros((num_train, num_classes))
    delta_W = np.zeros(W.shape)
    scores = np.zeros((num_train, num_classes))
    dh = 0.00000001
    
    
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :].T
                dW[:, y[i]] -= X[i, :].T
   
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W * 2
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_class = W.shape[1]
    num_image = X.shape[0]
    score = X.dot(W)
    delta = 1
    margin = np.zeros((num_image, num_class))
    D = W.shape[0]
    
    correct_class_score = score[np.arange(0, num_image), y]
    deltas = np.ones(score.shape)
    margin = score - correct_class_score.reshape((num_image, 1)) + deltas
    margin[margin < 0] = 0
    margin[np.arange(0, num_image), y] = 0
    loss = np.sum(margin)
    loss /= num_image
    loss += reg * np.sum(W * W)
    pass

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
   
    margin = score - correct_class_score.reshape((num_image, 1)) + deltas    
    margin[margin < 0] = 0
    margin[margin > 0] = 1
    margin[np.arange(0, num_image), y] = 0
    margin[np.arange(0, num_image), y] = -1 * np.sum(margin, axis = 1)
    dW = np.dot(X.T, margin)
    dW = dW / num_image + reg * W * 2
    pass
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
