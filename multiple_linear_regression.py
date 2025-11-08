import numpy as np

def compute_cost(X, y, w, b):
    """
    Computes the cost for linear regression using vectorized operations.

    """

    m = X.shape[0] # number of training examples
    cost = 0.0

    f_wb = X @ w + b # predicted value for the i-th example
    cost_sum = np.sum((f_wb - y) ** 2)    # sum of squared errors
    cost = (1 / (2 * m)) * cost_sum       # scaled cost
    
    return cost

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression using vectorized operations.
    
    """

    m = X.shape[0] # number of training examples

    err = (X @ w + b) - y # errors
    dj_dw = (X.T @ err) / m # weight gradients
    dj_db = np.sum(err) / m # bias gradient

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs gradient descent to learn w and b.

    Args:
      # X (ndarray (m,n)): Data, m examples with n features (no bias column; bias handled separately by b)
      y (ndarray (m,)) : Target values
      w_in, b_in       : Initial model parameters
      alpha (float)    : Learning rate
      num_iters (int)  : Number of iterations

    Returns:
      w, b, J_history
    """

    J_history = [] # to store cost at each iteration
    w = w_in.copy() # initial weights
    b = b_in # initial bias

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b) # compute gradients

        w = w - alpha * dj_dw # update weights
        b = b - alpha * dj_db # update bias

        if i < 100000: # limit history size
            cost = cost_function(X, y, w, b) # compute cost
            J_history.append(cost) # store cost

    return w, b, J_history
