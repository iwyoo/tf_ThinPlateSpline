import tensorflow as tf
import numpy as np
from PIL import Image
from ThinPlateSpline import ThinPlateSpline as stn

img = np.array(Image.open("original.png"))
out_size = list(img.shape)
shape = [1]+out_size+[1]

p = np.array([
  [-0.5, -0.5],
  [0.5, -0.5],
  [-0.5, 0.5],
  [0.5, 0.5]])

v = np.array([
  [0.2, 0.2],
  [0.4, 0.4],
  [0.6, 0.6],
  [0.8, 0.8]])

p = tf.constant(p.reshape([1, 4, 2]), dtype=tf.float32)
v = tf.constant(v.reshape([1, 4, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
t_img = stn(t_img, p, v, out_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  img1 = sess.run(t_img)
  Image.fromarray(np.uint8(img1.reshape(out_size))).save("transformed.png")
