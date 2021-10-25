import tensorflow as tf
from tf.keras.layers import Dense
from tf.keras.initializers import Constant


x = tf.constrant([10.])  # input setting
dense = Dense(units=1, activation='linear')  # imp. an affine function

print(dense.get_weights())

# Dense layer를 초기화할 때 weight나 bias의 크기에 대해 설정을 해 주지 않음.
# 따라서 이 상태에서 dense.get_weights()를 통해 weight와 bias를 받으려고 하면 빈 array를 반환할 것
# 이 dense layer의 weight와 bias는 아래처럼 실제로 첫 데이터를 넣어줄 때 그 크기를 통해 초기화가 진행됨

y_tf = dense(x)  # forward propagation + params initialization

W, B = dense.get_weights()

print('W: {}\n{}\n'.format(W.shapge, W))
print('B: {}\n{}\n'.format(B.shapge, B))



# weight/bias setting

w, b = tf.constant(10.), tf.constant(20.)
w_init, b_init = Constant(w), Constant(b)

dense = Dense(units=1,
              activation='linear',
              kernel_initializer=w_init,
              bias_initializer=b_init)

y_tf = dense(x)

W, B = dense.get_weights()

print('W: {}\n{}\n'.format(W.shapge, W))
print('B: {}\n{}\n'.format(B.shapge, B))



# Affine functions with n features

x = tf.random.uniform(shape=[1, 10], minval=0, maxval=10)
print(x.shape, '\n', x)


dense = Dense(units=1)
y_tf = dense(x)
W, B = dense.get_weights()

print('W: {}\n{}\n'.format(W.shapge, W))
print('B: {}\n{}\n'.format(B.shapge, B))
