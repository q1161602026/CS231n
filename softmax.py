import numpy as np

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), y]))
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  
  dS = softmax_output.copy()
  dS[xrange(num_train), y] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W 

  return loss, dW
