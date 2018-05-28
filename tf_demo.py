# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:32:32 2018

@author: Rithesh
"""

import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

x_input = np.array([[0,1,0,1],[0,0,1,1]])
y_input = np.array([[1,0,0,1]])

layer_dims = np.array([2,4,1])
activations = np.array(['relu','sigmoid'])
learning_rate = 0.01

tf.reset_default_graph()
np.random.seed(3)


def initialize_parameters(layer_dims):
    parameters = {}
    
    for layer in range(1,len(layer_dims)):
        parameters['W'+str(layer)] = tf.Variable(np.random.rand(layer_dims[layer],layer_dims[layer-1]),dtype='float32',name='W'+str(layer)) 
        parameters['b'+str(layer)] = tf.Variable(np.zeros((layer_dims[layer],1)),dtype='float32',name='b'+str(layer))
    
    return parameters


def activation_functions(x,activation_name):
    if (activation_name == 'relu'):
        out = tf.nn.relu(x)
    elif (activation_name == 'sigmoid'):
        out = tf.sigmoid(x)
    elif (activation_name == 'tanh'):
        out = tf.nn.tanh(x)
    
    return out

x = tf.placeholder(shape = x_input.shape,dtype='float32',name='x')
y = tf.placeholder(shape = y_input.shape,dtype='float32',name='y')

parameters = initialize_parameters(layer_dims)
z = {}
a = {}
n_layers = len(parameters)//2
a[str(0)] = x

for layer in range(1,n_layers+1):
    z[str(layer)] = tf.add(tf.matmul(parameters['W'+str(layer)],a[str(layer-1)]),parameters['b'+str(layer)])
    a[str(layer)] = activation_functions(z[str(layer)],activations[layer-1])


log_loss = (-1)*(y*tf.log(a[str(len(layer_dims)-1)]) + (1-y)*tf.log(1-a[str(len(layer_dims)-1)]))
#log_loss = tf.nn.softmax_cross_entropy_with_logits(logits = a[str(len(layer_dims)-1)], labels = y)
cost = tf.reduce_sum(log_loss)    
upd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)   
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2000):
    sess.run(upd, feed_dict = {x:x_input,y:y_input})
    cost_each_iteration = sess.run(cost,feed_dict = {x:x_input,y:y_input})
    output_at_last_iter = sess.run(a[str(len(layer_dims)-1)],feed_dict = {x:x_input,y:y_input})
    log_loss_at_last_iter = sess.run(log_loss,feed_dict = {x:x_input,y:y_input})


print (output_at_last_iter)