#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:47:27 2019

@author: bmw
"""
import tensorflow as tf
import numpy as np
hello = tf.constant("Hello")
print(hello)
sess = tf.Session()
print(sess.run(hello))
#hello.eval()

c = tf.Variable("Elephant",tf.string)
d= tf.zeros([10,20,20,2])
#print(tf.rank(c))

a = tf.constant(([2,3],[3,5],[4,6]),dtype=tf.float32)
b = tf.constant(([3,3,8],[1,1,2]),dtype=tf.float32)
c = tf.transpose(b)
#print(sess.run(c))
#print(sess.run(tf.matmul(b,c)))


d = tf.constant(([16,2,3,18],[15,9,10,2],[11,9,1,2],[1,19,23,0]),dtype=tf.float32)
f = tf.constant(([1,2,3,4]),dtype=tf.float32)
#g = print(sess.run(tf.matmul(tf.transpose(d),f)))
g = tf.math.log(f)

e = tf.constant(([1],[2],[3],[4]),dtype=tf.float32)
print(sess.run(tf.linalg.cross(f,e)))

#new = tf.reshape(e,[4,6])

#print(sess.run(e))
#print(sess.run(new))

x=tf.placeholder(tf.float32,shape=(2,3))
y = np.random.rand(2,3)
#print(sess.run(x, feed_dict={x:y}))

