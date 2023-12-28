import numpy as np
import math
from copy import deepcopy

#This implementation of Adams is from my work in programming assignment in AndrewNG's Coursera course

def update_parameters(params, grads, learning_rate):
    #dw = dJ/dw , db = dJ/db
    '''
    :param params: dictionary of parameters
        Here l is a layer
        param["W+l"] = Wl
        param["b+l"] = bl
    :param grads:
        grads["dW+l"] = dWl
        grads["db+l] = dbl
    :return: updated params
    '''

    #update params
    L = len(params)//2
    for l in range(1, L+1):
        params['W'+str(l)] = params['W'+str(l)] -learning_rate*grads["dW"+str(l)]
        params['b'+str(l)] = params["b"+ str(l)] - learning_rate*grads["db"+str(l)]


    return params

#formula
#   v_dw = beta1*v_dw +(1-beta1)*dw -momentum part
#   v_dw(corrected) = vdw/(1-beta1^t)
#   s_dw = beta2*s_dw + (1-beta2)*(dw)^2 - Adagrad part
#   s_dw(corrected) = s_dw/(1-beta2^t)
def initialize_adam(params):
    '''

    :param params:
    :return:
    '''

    L  = len(params) //2
    v = {}
    s = {}
    W_shape = parameters["W"+str(l)].shape
    b_shape = parameters["b"+str(l)].shape
    v["dW"+str(l)] = np.zeros(shape = (W_shape[0], W_shape[1]))
    s["dW"+str(l)] = np.zeros(shape = (W_shape[0], W_shape[1]))
    s["db"+str(l)] = np.zeros(shape = (b_shape[0], b_shape[1]))
    s["db"+str(l)] = np.zeros(shape = (b_shape[0], b_shape[1]))
