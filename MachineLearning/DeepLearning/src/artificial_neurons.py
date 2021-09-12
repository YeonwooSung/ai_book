import tensorflow as tf

from tf.math import exp, maximum
from tf.keras.layers import Activation, Dense


x = tf.random.normal(shape=[1, 5])

sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
relu = Activation('relu')

# forward propagation
y_sigmoid_tf = sigmoid(x)
y_tanh_tf = tanh(x)
y_relu_tf = relu(x)

print('Sigmoid(Tensorflow): {}\n{}\n'.format(y_sigmoid_tf.shape, y_sigmoid_tf.numpy()))
print('Tanh(Tensorflow): {}\n{}\n'.format(y_tanh_tf.shape, y_tanh_tf.numpy()))
print('ReLU(Tensorflow): {}\n{}\n'.format(y_relu_tf.shape, y_relu_tf.numpy()))



# Activation in Dense Layer

dense_sigmoid = Dense(units=1, activation='sigmoid')
y_sigmoid = dense_sigmoid(x)

print('AN with Sigmoid: {}\n{}'.format(y_sigmoid.shape, y_sigmoid.numpy()))
