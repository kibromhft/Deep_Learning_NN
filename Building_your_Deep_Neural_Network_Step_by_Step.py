#!/usr/bin/env python
# coding: utf-8

# # Building your Deep Neural Network: Step by Step
# 
# Welcome to your week 4 assignment (part 1 of 2)! Previously you trained a 2-layer Neural Network with a single hidden layer. This week, you will build a deep neural network with as many layers as you want!
# 
# - In this notebook, you'll implement all the functions required to build a deep neural network.
# - For the next assignment, you'll use these functions to build a deep neural network for image classification.
# 
# **By the end of this assignment, you'll be able to:**
# 
# - Use non-linear units like ReLU to improve your model
# - Build a deeper neural network (with more than 1 hidden layer)
# - Implement an easy-to-use neural network class
# 
# **Notation**:
# - Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer. 
#     - Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
# - Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example. 
#     - Example: $x^{(i)}$ is the $i^{th}$ training example.
# - Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
#     - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
# 
# Let's get started!

# ## Table of Contents
# - [1 - Packages](#1)
# - [2 - Outline](#2)
# - [3 - Initialization](#3)
#     - [3.1 - 2-layer Neural Network](#3-1)
#         - [Exercise 1 - initialize_parameters](#ex-1)
#     - [3.2 - L-layer Neural Network](#3-2)
#         - [Exercise 2 - initialize_parameters_deep](#ex-2)
# - [4 - Forward Propagation Module](#4)
#     - [4.1 - Linear Forward](#4-1)
#         - [Exercise 3 - linear_forward](#ex-3)
#     - [4.2 - Linear-Activation Forward](#4-2)
#         - [Exercise 4 - linear_activation_forward](#ex-4)
#     - [4.3 - L-Layer Model](#4-3)
#         - [Exercise 5 - L_model_forward](#ex-5)
# - [5 - Cost Function](#5)
#     - [Exercise 6 - compute_cost](#ex-6)
# - [6 - Backward Propagation Module](#6)
#     - [6.1 - Linear Backward](#6-1)
#         - [Exercise 7 - linear_backward](#ex-7)
#     - [6.2 - Linear-Activation Backward](#6-2)
#         - [Exercise 8 - linear_activation_backward](#ex-8)
#     - [6.3 - L-Model Backward](#6-3)
#         - [Exercise 9 - L_model_backward](#ex-9)
#     - [6.4 - Update Parameters](#6-4)
#         - [Exercise 10 - update_parameters](#ex-10)

# <a name='1'></a>
# ## 1 - Packages
# 
# First, import all the packages you'll need during this assignment. 
# 
# - [numpy](www.numpy.org) is the main package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - dnn_utils provides some necessary functions for this notebook.
# - testCases provides some test cases to assess the correctness of your functions
# - np.random.seed(1) is used to keep all the random function calls consistent. It helps grade your work. Please don't change the seed! 

# In[2]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from public_tests import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# <a name='2'></a>
# ## 2 - Outline
# 
# To build your neural network, you'll be implementing several "helper functions." These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. 
# 
# Each small helper function will have detailed instructions to walk you through the necessary steps. Here's an outline of the steps in this assignment:
# 
# - Initialize the parameters for a two-layer network and for an $L$-layer neural network
# - Implement the forward propagation module (shown in purple in the figure below)
#      - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
#      - The ACTIVATION function is provided for you (relu/sigmoid)
#      - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#      - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
# - Compute the loss
# - Implement the backward propagation module (denoted in red in the figure below)
#     - Complete the LINEAR part of a layer's backward propagation step
#     - The gradient of the ACTIVATE function is provided for you(relu_backward/sigmoid_backward) 
#     - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
#     - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# - Finally, update the parameters
# 
# <img src="images/final outline.png" style="width:800px;height:500px;">
# <caption><center><b>Figure 1</b></center></caption><br>
# 
# 
# **Note**:
# 
# For every forward function, there is a corresponding backward function. This is why at every step of your forward module you will be storing some values in a cache. These cached values are useful for computing gradients. 
# 
# In the backpropagation module, you can then use the cache to calculate the gradients. Don't worry, this assignment will show you exactly how to carry out each of these steps! 

# <a name='3'></a>
# ## 3 - Initialization
# 
# You will write two helper functions to initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one generalizes this initialization process to $L$ layers.
# 
# <a name='3-1'></a>
# ### 3.1 - 2-layer Neural Network
# 
# <a name='ex-1'></a>
# ### Exercise 1 - initialize_parameters
# 
# Create and initialize the parameters of the 2-layer neural network.
# 
# **Instructions**:
# 
# - The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
# - Use this random initialization for the weight matrices: `np.random.randn(shape)*0.01` with the correct shape
# - Use zero initialization for the biases: `np.zeros(shape)`

# In[3]:


# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    # YOUR CODE STARTS HERE #(≈ 4 lines of code)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 =  np.zeros((n_y,1))
            
    # YOUR CODE ENDS HERE
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


# In[4]:


parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test(initialize_parameters)


# ***Expected output***
# ```
# W1 = [[ 0.01624345 -0.00611756 -0.00528172]
#  [-0.01072969  0.00865408 -0.02301539]]
# b1 = [[0.]
#  [0.]]
# W2 = [[ 0.01744812 -0.00761207]]
# b2 = [[0.]]
# ```

# <a name='3-2'></a>
# ### 3.2 - L-layer Neural Network
# 
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep` function, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. For example, if the size of your input $X$ is $(12288, 209)$ (with $m=209$ examples) then:
# 
# <table style="width:100%">
#     <tr>
#         <td>  </td> 
#         <td> <b>Shape of W</b> </td> 
#         <td> <b>Shape of b</b>  </td> 
#         <td> <b>Activation</b> </td>
#         <td> <b>Shape of Activation</b> </td> 
#     <tr>
#     <tr>
#         <td> <b>Layer 1</b> </td> 
#         <td> $(n^{[1]},12288)$ </td> 
#         <td> $(n^{[1]},1)$ </td> 
#         <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td> 
#         <td> $(n^{[1]},209)$ </td> 
#     <tr>
#     <tr>
#         <td> <b>Layer 2</b> </td> 
#         <td> $(n^{[2]}, n^{[1]})$  </td> 
#         <td> $(n^{[2]},1)$ </td> 
#         <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td> 
#         <td> $(n^{[2]}, 209)$ </td> 
#     <tr>
#        <tr>
#         <td> $\vdots$ </td> 
#         <td> $\vdots$  </td> 
#         <td> $\vdots$  </td> 
#         <td> $\vdots$</td> 
#         <td> $\vdots$  </td> 
#     <tr>  
#    <tr>
#        <td> <b>Layer L-1</b> </td> 
#         <td> $(n^{[L-1]}, n^{[L-2]})$ </td> 
#         <td> $(n^{[L-1]}, 1)$  </td> 
#         <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td> 
#         <td> $(n^{[L-1]}, 209)$ </td> 
#    <tr>
#    <tr>
#        <td> <b>Layer L</b> </td> 
#         <td> $(n^{[L]}, n^{[L-1]})$ </td> 
#         <td> $(n^{[L]}, 1)$ </td>
#         <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
#         <td> $(n^{[L]}, 209)$  </td> 
#     <tr>
# </table>
# 
# Remember that when you compute $W X + b$ in python, it carries out broadcasting. For example, if: 
# 
# $$ W = \begin{bmatrix}
#     w_{00}  & w_{01} & w_{02} \\
#     w_{10}  & w_{11} & w_{12} \\
#     w_{20}  & w_{21} & w_{22} 
# \end{bmatrix}\;\;\; X = \begin{bmatrix}
#     x_{00}  & x_{01} & x_{02} \\
#     x_{10}  & x_{11} & x_{12} \\
#     x_{20}  & x_{21} & x_{22} 
# \end{bmatrix} \;\;\; b =\begin{bmatrix}
#     b_0  \\
#     b_1  \\
#     b_2
# \end{bmatrix}\tag{2}$$
# 
# Then $WX + b$ will be:
# 
# $$ WX + b = \begin{bmatrix}
#     (w_{00}x_{00} + w_{01}x_{10} + w_{02}x_{20}) + b_0 & (w_{00}x_{01} + w_{01}x_{11} + w_{02}x_{21}) + b_0 & \cdots \\
#     (w_{10}x_{00} + w_{11}x_{10} + w_{12}x_{20}) + b_1 & (w_{10}x_{01} + w_{11}x_{11} + w_{12}x_{21}) + b_1 & \cdots \\
#     (w_{20}x_{00} + w_{21}x_{10} + w_{22}x_{20}) + b_2 &  (w_{20}x_{01} + w_{21}x_{11} + w_{22}x_{21}) + b_2 & \cdots
# \end{bmatrix}\tag{3}  $$
# 

# <a name='ex-2'></a>
# ### Exercise 2 -  initialize_parameters_deep
# 
# Implement initialization for an L-layer Neural Network. 
# 
# **Instructions**:
# - The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# - Use random initialization for the weight matrices. Use `np.random.randn(shape) * 0.01`.
# - Use zeros initialization for the biases. Use `np.zeros(shape)`.
# - You'll store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for last week's Planar Data classification model would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! 
# - Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
# ```python
#     if L == 1:
#         parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#         parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
# ```

# In[5]:


# GRADED FUNCTION: initialize_parameters_deep

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        # YOUR CODE STARTS HERE #(≈ 2 lines of code)
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        # YOUR CODE ENDS HERE
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# In[6]:


parameters = initialize_parameters_deep([5,4,3])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_deep_test(initialize_parameters_deep)

# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    #(≈ 1 line of code)
    # Z = ...
    # YOUR CODE STARTS HERE
    Z = np.dot(W,A)+b
    # YOUR CODE ENDS HERE
    cache = (A, W, b)
    
    return Z, cache


# In[8]:


t_A, t_W, t_b = linear_forward_test_case()
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
print("Z = " + str(t_Z))

linear_forward_test(linear_forward)


# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z , linear_cache = linear_forward(A_prev, W, b) # Linear activation
        A , activation_cache = sigmoid(Z)    # Non-linear activation)
        # YOUR CODE ENDS HERE
    
    elif activation == "relu":
        
        # YOUR CODE STARTS HERE #(≈ 2 lines of code)
        Z , linear_cache = linear_forward(A_prev, W, b)
        A , activation_cache = relu(Z)
        # YOUR CODE ENDS HERE
    cache = (linear_cache, activation_cache)

    return A, cache


# In[10]:


t_A_prev, t_W, t_b = linear_activation_forward_test_case()

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid")
print("With sigmoid: A = " + str(t_A))

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "relu")
print("With ReLU: A = " + str(t_A))

linear_activation_forward_test(linear_activation_forward)



# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A 
        # A, cache = ...
        # caches ...
        # YOUR CODE STARTS HERE  #(≈ 2 lines of code)
        A , cache =  linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        # YOUR CODE ENDS HERE
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    #(≈ 2 lines of code)
    # AL, cache = ...
    # caches ...
    # YOUR CODE STARTS HERE
    AL ,  cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    # YOUR CODE ENDS HERE
          
    return AL, caches


# In[12]:


t_X, t_parameters = L_model_forward_test_case_2hidden()
t_AL, t_caches = L_model_forward(t_X, t_parameters)

print("AL = " + str(t_AL))

L_model_forward_test(L_model_forward)


# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    # (≈ 1 lines of code)
    # cost = ...
    # YOUR CODE STARTS HERE
    cost = -1 / m * np.sum(np.multiply(np.log(AL) , Y) + np.multiply((1-Y) , np.log(1-AL)))
    # YOUR CODE ENDS HERE
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    
    return cost


# In[16]:


t_Y, t_AL = compute_cost_test_case()
t_cost = compute_cost(t_AL, t_Y)

print("Cost: " + str(t_cost))

compute_cost_test(compute_cost)


A = np.array([[1, 2], [3, 4]])

print('axis=1 and keepdims=True')
print(np.sum(A, axis=1, keepdims=True))
print('axis=1 and keepdims=False')
print(np.sum(A, axis=1, keepdims=False))
print('axis=0 and keepdims=True')
print(np.sum(A, axis=0, keepdims=True))
print('axis=0 and keepdims=False')
print(np.sum(A, axis=0, keepdims=False))



# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    # YOUR CODE STARTS HERE
    dW = 1/m * np.dot(dZ , A_prev.T)
    db = 1/m * np.sum(dZ , axis = 1 , keepdims = True)
    dA_prev = np.dot(W.T , dZ)
    
    # YOUR CODE ENDS HERE
    
    return dA_prev, dW, db


# In[19]:


t_dZ, t_linear_cache = linear_backward_test_case()
t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)

print("dA_prev: " + str(t_dA_prev))
print("dW: " + str(t_dW))
print("db: " + str(t_db))

linear_backward_test(linear_backward)

# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        # YOUR CODE STARTS HERE
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db= linear_backward(dZ, linear_cache)
        
        # YOUR CODE ENDS HERE
        
    elif activation == "sigmoid":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        # YOUR CODE STARTS HERE
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db= linear_backward(dZ, linear_cache)
        
        # YOUR CODE ENDS HERE
    
    return dA_prev, dW, db


# In[21]:


t_dAL, t_linear_activation_cache = linear_activation_backward_test_case()

t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "sigmoid")
print("With sigmoid: dA_prev = " + str(t_dA_prev))
print("With sigmoid: dW = " + str(t_dW))
print("With sigmoid: db = " + str(t_db))

t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "relu")
print("With relu: dA_prev = " + str(t_dA_prev))
print("With relu: dW = " + str(t_dW))
print("With relu: db = " + str(t_db))

linear_activation_backward_test(linear_activation_backward)

# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    #(1 line of code)
    # dAL = ...
    # YOUR CODE STARTS HERE
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
    # YOUR CODE ENDS HERE
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    #(approx. 5 lines)
    # current_cache = ...
    # dA_prev_temp, dW_temp, db_temp = ...
    # grads["dA" + str(L-1)] = ...
    # grads["dW" + str(L)] = ...
    # grads["db" + str(L)] = ...
    # YOUR CODE STARTS HERE
    current_cache = caches[L-1]   
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,current_cache , "sigmoid")
    grads["dA" + str(L-1)] =  dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # YOUR CODE ENDS HERE
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        #(approx. 5 lines)
        # current_cache = ...
        # dA_prev_temp, dW_temp, db_temp = ...
        # grads["dA" + str(l)] = ...
        # grads["dW" + str(l + 1)] = ...
        # grads["db" + str(l + 1)] = ...
        # YOUR CODE STARTS HERE
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp,current_cache , "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        # YOUR CODE ENDS HERE

    return grads


# In[25]:


t_AL, t_Y_assess, t_caches = L_model_backward_test_case()
grads = L_model_backward(t_AL, t_Y_assess, t_caches)

print("dA0 = " + str(grads['dA0']))
print("dA1 = " + str(grads['dA1']))
print("dW1 = " + str(grads['dW1']))
print("dW2 = " + str(grads['dW2']))
print("db1 = " + str(grads['db1']))
print("db2 = " + str(grads['db2']))

L_model_backward_test(L_model_backward)


# GRADED FUNCTION: update_parameters

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop. #(≈ 2 lines of code)
    
    for l in range(L):
        # parameters["W" + str(l+1)] = ...
        # parameters["b" + str(l+1)] = ...
        # YOUR CODE STARTS HERE
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate *  grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate *  grads["db" + str(l + 1)]
        # YOUR CODE ENDS HERE
    return parameters


# In[27]:


t_parameters, grads = update_parameters_test_case()
t_parameters = update_parameters(t_parameters, grads, 0.1)

print ("W1 = "+ str(t_parameters["W1"]))
print ("b1 = "+ str(t_parameters["b1"]))
print ("W2 = "+ str(t_parameters["W2"]))
print ("b2 = "+ str(t_parameters["b2"]))

update_parameters_test(update_parameters)


# **Expected output:**
# 
# ```
# W1 = [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
#  [-1.76569676 -0.80627147  0.51115557 -1.18258802]
#  [-1.0535704  -0.86128581  0.68284052  2.20374577]]
# b1 = [[-0.04659241]
#  [-1.28888275]
#  [ 0.53405496]]
# W2 = [[-0.55569196  0.0354055   1.32964895]]
# b2 = [[-0.84610769]]
# ```

# ### Congratulations! 
# 
# You've just implemented all the functions required for building a deep neural network, including: 
# 
# - Using non-linear units improve your model
# - Building a deeper neural network (with more than 1 hidden layer)
# - Implementing an easy-to-use neural network class
# 
# This was indeed a long assignment, but the next part of the assignment is easier. ;) 
# 
# In the next assignment, you'll be putting all these together to build two models:
# 
# - A two-layer neural network
# - An L-layer neural network
# 
# You will in fact use these models to classify cat vs non-cat images! (Meow!) Great work and see you next time. 
