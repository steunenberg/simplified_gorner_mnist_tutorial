# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

RUNS = 10001
# prepare status updates, as the whole excercise may take quite some time
displays = 40 # how many time do you want to see the status
display_trigger = int(RUNS/displays)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

df = pd.DataFrame(columns=['train_accuracy', 'test_accuracy','train_cross_entropy', 'test_cross_entropy'])

# learning rate decay, preparation
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations

for i in range(RUNS):
    if (i % display_trigger) == 0:
        print(" run #" + str(i) + " of " + str(RUNS) + " runs.") 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X:batch_X, Y_: batch_Y, lr: learning_rate}
    
    sess.run(train_step, feed_dict=train_data)
    
    #success?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    
    # success on test data?
    test_data = {X : mnist.test.images, Y_ : mnist.test.labels}
    at, ct = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    
    df.loc[len(df)] = [a, at, c, ct]
    
# display results
# display results
df1 = df[['train_accuracy', 'test_accuracy']]
df2 = df[['train_cross_entropy', 'test_cross_entropy']]
df1.plot()
df2.plot()
plt.show()

    
    
    
# layers 4 8 12 200, patches 5x5str1 5x5str2 4x4str2 best 0.989 after 10000 iterations
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 best 0.9892 after 10000 iterations
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 best 0.9908 after 10000 iterations but going downhill from 5000 on
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75 best 0.9922 after 10000 iterations (but above 0.99 after 1400 iterations only)
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9914 at 13700 iterations
# layers 9 16 25 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9918 at 10500 (but 0.99 at 1500 iterations already, 0.9915 at 5800)
# layers 9 16 25 300, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9916 at 5500 iterations (but 0.9903 at 1200 iterations already)
# attempts with 2 fully-connected layers: no better 300 and 100 neurons, dropout 0.75 and 0.5, 6x6 5x5 4x4 patches no better
#*layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 dropout=0.75 best 0.9928 after 12800 iterations (but consistently above 0.99 after 1300 iterations only, 0.9916 at 2300 iterations, 0.9921 at 5600, 0.9925 at 20000)
# layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 no dropout best 0.9906 after 3100 iterations (avove 0.99 from iteration 1400)
