import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes=W.shape[1] # C
  num_train=X.shape[0] # N

  for i in range(num_train): # i = 0 to N
    scores=np.dot(X[i], W) # s_0 ~ s_C; range [0,1]
    scores_exp=np.exp(scores)
    correct_exp=scores_exp[y[i]]  
    probs=scores_exp/np.sum(scores_exp)

    loss += -np.log(correct_exp/np.sum(scores_exp))
    
    for j in range(num_classes):
      dW[:,j] += X[i] * probs[j]
    dW[:,y[i]]-= X[i]
  loss = loss / num_train
  dW /= num_train

  loss += reg * np.sum(np.matmul(W.T, W))
  dW += 2*reg*W

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

  num_classes=W.shape[1] # C
  num_train=X.shape[0] # N
  
  scores=np.dot(X, W) # N x C
  scores_exp=np.exp(scores)
  correct_exp=scores_exp[list(range(num_train)), y]
  probs=scores_exp/np.sum(scores_exp, axis=1, keepdims=True)
  
  loss = -np.sum(np.log(correct_exp / np.sum(scores_exp, axis=1, keepdims=True).T))
  loss /= num_train
  loss += reg * np.sum(np.matmul(W.T, W))
  

  #scores_exp를 scores_exp의 summation으로 각각 나누어준다.
  probs[list(range(num_train)), y] -= 1.0
  dW = np.dot(X.T, probs)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

