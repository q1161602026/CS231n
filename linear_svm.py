import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  return loss, dW
  """
  Structured SVM loss function, vectorized implementation.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[xrange(num_train), y].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[xrange(num_train), y] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

  binary = np.zeros((num_train, num_classes))
  binary[margins > 0] = 1
  binary[range(num_train), y] = -np.sum(binary, axis=1)
  dW = (X.T).dot(binary)
  dW = dW/num_train + reg*W

  return loss, dW
