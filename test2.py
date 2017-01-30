import tensorflow as tf
import numpy as np
from PIL import Image
from ThinPlateSpline2 import ThinPlateSpline2 as stn

img = np.array(Image.open("original.png"))
out_size = list(img.shape)
shape = [1]+out_size+[1]

s_ = np.array([ # source position
  [-0.5, -0.5],
  [0.5, -0.5],
  [-0.5, 0.5],
  [0.5, 0.5]])

t_ = np.array([ # target position
  [-0.3, -0.3],
  [0.3, -0.3],
  [-0.3, 0.3],
  [0.3, 0.3]])

s = tf.constant(s_.reshape([1, 4, 2]), dtype=tf.float32)
t = tf.constant(t_.reshape([1, 4, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
t_img = stn(t_img, s, t, out_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  img1 = sess.run(t_img)
  Image.fromarray(np.uint8(img1.reshape(out_size))).save("transformed2.png")
