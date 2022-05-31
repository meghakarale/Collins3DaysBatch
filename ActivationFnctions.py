# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:03:45 2022
Activation Functions
@author: TSE
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))      # squeezing nature , it returns value in the range 0 to 1 only
                                # vanishing Gradient problem

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
                                # squeezing nature , it returns value in the range -1 to 1 only
                                    # vanishing Gradient problem
                                    
def relu(x):
    return np.maximum(0,x)
                                    

def leakyR(x):
    return np.maximum(0.01*x,x)

def elu(x):
    return np.maximum(x, 0.1*(np.exp(x)-1))

InputX_W = np.arange(-5.0,5.0,0.1)

y_sigmoid = sigmoid(InputX_W)
y_tanh = tanh(InputX_W)
y_relu = relu(InputX_W)
y_leaky = leakyR(InputX_W)
y_elu = elu(InputX_W)

fig,axes = plt.subplots(ncols=5,figsize=(20,5))

ax = axes[0]
ax.plot(InputX_W, y_sigmoid)
ax.set_ylim([-0.5,1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('InputX_W')
ax.set_title('Sigmoid function')

ax = axes[1]
ax.plot(InputX_W, y_tanh)
ax.set_ylim([-1.,1])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('InputX_W')
ax.set_title('Tanh function')

ax = axes[2]
ax.plot(InputX_W, y_relu)
ax.set_ylim([-0.5,1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('InputX_W')
ax.set_title('ReLu function')

ax = axes[3]
ax.plot(InputX_W, y_leaky)
ax.set_ylim([-0.5,1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('InputX_W')
ax.set_title('Leaky ReLu ')


ax = axes[4]
ax.plot(InputX_W, y_elu)
ax.set_ylim([-0.5,1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('InputX_W')
ax.set_title('Exponential ReLu ')



