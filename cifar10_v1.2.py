#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:24:37 2017

to do:
    -operation names
    -save  & restore model
    -evaluate model
    -tensorboard

@author: sabpap
"""

##############################################################################
"""
Image Classification on tensorflow
Dataset: cifar10
"""
##############################################################################

from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy as np
import tensorflow as tf

##############################################################################
"""
Create Helper Functions/Classes:
    
-FeedNextBatch() #custom??
-conv2d()
-maxpool2d()

"""
#full conv layer operation
def conv2d(x, W, b, strides=1,pad='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding= pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

#maxpool operation
def maxpool2d(x, k=2,pad='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
padding=pad)

#custom dataset class + feed_next_batch function 
class cifar10_data(object):
    def __init__(self):
        #import cifar10 from keras datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        
        #splitting test data to validation set and test set
        (self.x_validation, self.y_validation ) = (self.x_test[0:5000] , self.y_test[0:5000])
        (self.x_test, self.y_test) = (self.x_test[5000:] , self.y_test[5000:])
        
        
        #Dataset features
        self.num_train = self.x_train.shape[0] 
        self.num_validation = self.x_validation.shape[0]
        self.num_test = self.x_test.shape[0]
        self.num_classes = np.max(self.y_train)+1
        self.im_rows = self.x_train.shape[1]
        self.im_cols = self.x_train.shape[2]
        self.color_channels = self.x_train.shape[3]
        
        #rescale data
        self.x_train = self.x_train.astype('float32') 
        self.x_train /= 255
        self.x_validation = self.x_validation.astype('float32') 
        self.x_validation /= 255
        self.x_test = self.x_test.astype('float32') 
        self.x_test /= 255
        
        #one-hot encode labels
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.y_validation = to_categorical(self.y_validation)
        
        #Dataset Summary
        print("\nCIFAR10 loaded..\n")
        print("-%d training examples" % self.num_train)
        print("-%d validation examples" % self.num_validation)
        print("-%d test examples\n" % self.num_test)
        print("-input image shape : (%d,%d,%d) " %(self.im_rows,self.im_cols,self.color_channels))
        
        self.pointer = 0
        self.current_epoch = 0
        self.current_iter = 0
        
    def feed_next_batch(self, state, batch_size=32, reset=False):
         
        
        #reset values if reset == True
        if reset == True:
            self.pointer = 0
            self.current_epoch = 0
            self.current_iter = 0
        
        self.current_iter += 1
        
        if state == 'train':
            pointer_end = self.x_train.shape[0] // batch_size
        elif state == 'validation':    
            pointer_end = self.x_validation.shape[0] // batch_size
        elif state == 'test':
            pointer_end = self.x_test.shape[0] // batch_size
        else:
                raise NameError("pass state: train or validation or test ")
            
        if self.pointer ==  pointer_end:
            if state == 'train':
                nextX_batch = self.x_train[self.pointer*batch_size:]
                nextY_batch = self.y_train[self.pointer*batch_size:]
            elif state == 'validation':
                nextX_batch = self.x_validation[self.pointer*batch_size:]
                nextY_batch = self.y_validation[self.pointer*batch_size:]
            elif state == 'test':
                nextX_batch = self.x_test[self.pointer*batch_size:]
                nextY_batch = self.y_test[self.pointer*batch_size:]
            else:
                raise NameError("set state: train or validation or test ")
            self.pointer = 0
            self.current_epoch += 1
        else:
            if state == 'train':
                nextX_batch = self.x_train[self.pointer*batch_size:(self.pointer+1)*batch_size]
                nextY_batch = self.y_train[self.pointer*batch_size:(self.pointer+1)*batch_size]
            elif state == 'validation':
                nextX_batch = self.x_validation[self.pointer*batch_size:(self.pointer+1)*batch_size]
                nextY_batch = self.y_validation[self.pointer*batch_size:(self.pointer+1)*batch_size]
            elif state == 'test':
                nextX_batch = self.x_test[self.pointer*batch_size:(self.pointer+1)*batch_size]
                nextY_batch = self.y_test[self.pointer*batch_size:(self.pointer+1)*batch_size]
            else:
                raise NameError("set state: train or validation or test ")
            
            self.pointer += 1    
        return nextX_batch, nextY_batch
##############################################################################

##############################################################################
""" Load CIFAR10 Dataset """

#create object of custom cifar10 class
dataset = cifar10_data()
##############################################################################

##############################################################################
""" Set Parameters """

batch_size = 64
learning_rate = 0.0001    
training_iters = 100000
display_step = 10
dropout = 0.5
kernel_size = 3

##############################################################################

##############################################################################
"""Build Model"""

tf.reset_default_graph()
graph = tf.get_default_graph()

# tf Graph input
x = tf.placeholder(tf.float32, [None, dataset.im_rows, dataset.im_cols, dataset.color_channels], name ='x_train_batch')
y = tf.placeholder(tf.float32, [None, dataset.num_classes] , name='y_train_batch')
drop = tf.placeholder(tf.float32) #dropout 

### tf Variables(weights+biases) ###

#conv layer #1
w1 = tf.get_variable('w1', shape=[kernel_size,kernel_size,dataset.color_channels,32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1', shape=[32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

#conv layer #2
w2 = tf.get_variable('w2', shape=[kernel_size,kernel_size,32,64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
b2 = tf.get_variable('b2', shape=[64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

#conv layer #3
w3 = tf.get_variable('w3', shape=[kernel_size,kernel_size,64,128],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
b3 = tf.get_variable('b3', shape=[128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

#fcl layer #4
w4 = tf.get_variable('w4', shape=[4*4*128,1024],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
b4 = tf.get_variable('b4', shape=[1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

#fcl layer #4
w5 = tf.get_variable('w5', shape=[1024,dataset.num_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01))
b5 = tf.get_variable('b5', shape=[dataset.num_classes], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

### tf Operations ###

#Layer 1 input:(batch_size, 32, 32, 3) ---> output (batch_size, 16, 16, 32)
l1_out = conv2d(x, w1, b1, strides=1) #convlayer 1 output
mp1_out = maxpool2d(l1_out) #maxPoolLayer 1 output

#Layer 2 input:(batch_size, 16, 16, 32) ---> output (batch_size, 8, 8, 64)
l2_out = conv2d(mp1_out, w2, b2, strides=1) #convlayer 2 output
mp2_out = maxpool2d(l2_out) #maxPoolLayer 2 output

#Layer 3 input:(batch_size, 8, 8, 64) ---> output (batch_size, 4, 4, 128)
l3_out = conv2d(mp2_out, w3, b3, strides=1) #convlayer 3 output
mp3_out = maxpool2d(l3_out) #maxPoolLayer 2 output


#flatten
fl3 = tf.contrib.layers.flatten(mp3_out)


#Layer 4 input:(batch_size, 4, 4, 128) ---> output (batch_size, 4, 4, 1024)
l4_out = tf.add(tf.matmul(fl3, w4),b4) #convlayer 4(fully connected layer) output
l4_out2 = tf.nn.relu(l4_out)

 # Apply Dropout
l4_out2 = tf.nn.dropout(l4_out2, dropout)

#Layer 5 input:(batch_size, 4, 4, 128) ---> output (batch_size, 4, 4, 10)
out_1d = tf.add(tf.matmul(l4_out2, w5), b5) #convlayer 4(fully connected layer) output



### Final operations ####

#cost function
cost = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = out_1d)
cost = tf.reduce_mean(cost)
tf.identity(cost,name='cost')
#predictions
predictions = tf.argmax(out_1d,1) #model's predictions for current batch
true_classes = tf.argmax(y,1) #correct classes for current batch
correct_predictions = tf.equal(predictions,true_classes) #model's correct predictions for current batch

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) #models accuracy on current batch

#optimizer algorithm
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initialize graphs variables at this point
init = tf.global_variables_initializer()
##############################################################################

##############################################################################
""" Train Model """ 



with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iters):
        current_batch = dataset.feed_next_batch(state='train',batch_size=batch_size,reset=False)
        cost_python, accuracy_python, _ = sess.run([cost, accuracy, optimizer], feed_dict = {x:current_batch[0], y:current_batch[1], drop:dropout})
        print 'Cost in step %d is %f' %(dataset.current_iter, cost_python)
        print 'Accuracy in training batch %d is %f' %(dataset.current_iter, accuracy_python)
        
        """if i % display_step == 0:
            accuracy_python = sess.run(accuracy, feed_dict = {x:dataset.x_validation, y:dataset.y_validation})
            print 'Accuracy in validation set is %f' %(accuracy_python)
            
     
