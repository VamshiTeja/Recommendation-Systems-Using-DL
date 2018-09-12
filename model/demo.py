# -*- coding: utf-8 -*-
"""
| **@created on:** 11/05/17,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| DL Module Tests
| **Sphinx Documentation Status:** Complete
|
Aim:
1. Write Custom Operation in Tensorflow
2. Write Custom Gradient Function for the Custom Operation
Example:
    The model calculates a simple euclidean distance between two vectors (x,y) with an addition of a weight which is
    previous added to our training label. The objective of the model is to come up with the right constant which was
    added during data generation. In this case the constant added is 23
"""

import random
import tensorflow as tf
random.seed(0)
import math
import numpy as np
from tensorflow.python.framework import ops

# Data Generation

# Customers
SAMPLES = 100

# The constant which we expect the network to learn
CONSTANT = 23

# Number of Iterations
epoch = 10000

# Learning Rate
learning_rate = 0.8

"""
Training Data
Data: [[6.0, 3.0], ....]
Label: [[28.19]]
Equations: sqrt(x^2 - y^2) + CONSTANT
"""
train_data = [[random.uniform(5, 10), random.uniform(1, 5)] for i in range(SAMPLES)]
train_label = [[math.sqrt((x ** 2) - (y ** 2)) + CONSTANT] for x, y in train_data]


# Python Custom Gradient Function
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate Random Gradient name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad' + 'ABC@a1b2c3'

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# Our Custom Euclidean Distance Function
def eu_dist(x, y, z, name=None):
    """
    Custom Function which defines pyfunc and gradient override
    :param x: X
    :param y: Y
    :param z: Z
    :param name: Function name
    :return: Equation Output, Calculated Gradient
    """
    with ops.name_scope(name, "EuDist", [x, y, z]) as name:
        """
        Our pyfunc accepts 3 input parameters and returns 2 outputs
        Input Parameters: x, y, z
        Output Parameters: euclidean distance equation, euclidean distance prime equation(d/dx)
        """
        eud, grad = py_func(eu_dist_grad,
                            [x, y, z],
                            [tf.float32, tf.float32],
                            name=name,
                            grad=_EuDistGrads)
        return eud, grad


# Core Function used for pyfunc
def eu_dist_grad(x, y, z):
    """
    Euclidean Distance Equation
    :param x: X
    :param y: Y
    :param z: Z
    :return: equation, equation_prime
    """
    # Euclidean Distance with a weight
    equation = np.sqrt((x ** 2) - (y ** 2)) + z

    # d/dx (Euclidean Distance)
    equation_prime = x / np.sqrt((x ** 2) - (y ** 2)) + z
    return equation, equation_prime


# Our Gradient Function
def _EuDistGrads(op, grads, grad_glob):
    """
    Custom Gradient Function
    :param op:  Operation - operation.inputs = [x,y,z], operation.outputs=[equation, equation_prime]
    :param grads: Gradients for equation prime
    :param grad_glob: - No real use of it, but the gradient function parameter size should match op.inputs
    :return: x ,y, z*grads
    """
    return op.inputs[0], op.inputs[1], op.inputs[2]*grads


# Declare Placeholders

# Input Placeholder for training data
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# Output Placeholder for training label
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# Custom variable to be used in euclidean distance - Optimizee Variable
z = tf.Variable(tf.random_uniform(shape=[SAMPLES, 1]), dtype='float32', trainable=True)


def model(model_inp):
    """
    Simple Model
    :param model_inp: Model Input [[x,y], .....]
    :return: Dictionary of variables
    """

    # Split model_input into x and y based on columns
    x, y = tf.split(model_inp, 2, axis=1)
    # Call custom pyfunc with gradient mapping
    ed_layer, grad = eu_dist(x, y, z)
    # Return dictionary of variables
    return {'ed': ed_layer, 'addvar': z, 'grad': grad}


# Model Declaration
pred = model(x)
# Mean Square Error as our objective function
cost = tf.reduce_mean(tf.reduce_mean(tf.square(y - pred['ed'])))
# AdaGrad Optimizer used to minimize above objective function
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# Summary writer to writer generated graph
writer = tf.summary.FileWriter(logdir='/tmp/dlp1/', graph=tf.get_default_graph())
# Global Variable initializer
init = tf.global_variables_initializer()

# Run the graph
with tf.Session() as sess:
    # Run Initializer
    sess.run(init)

    # Epoch Training
    for e in range(epoch):
        c, _, ed, av, g = sess.run([cost, optimizer, pred['ed'], pred['addvar'], pred['grad']], feed_dict={
            x: train_data,
            y: train_label,
        })
        print('Epoch: ', e, ' Cost: ', c, 'addvar', av[0], 'grad:', g[0])