#%% LOAD TO TFRECORDS
# Loading necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.chdir("D:/MachineLearning/Algorithms/NeuralNetworks/tfrecords")

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0

# Track the data type
dataType = x_train.dtype
print(f"Data type: {dataType}")
labelType = y_test.dtype
print(f"Data type: {labelType}")

# Visualization
im_list = []
n_samples_to_show = 16
c = 0
for i in range(n_samples_to_show):
  im_list.append(x_train[i])

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, im_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, 'gray')
plt.show()

# Convert values to compatible tf.Example types.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create the features dictionary.
def image_example(image, label, dimension):
    feature = {
        'dimension': _int64_feature(dimension),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#Create tfrecords
record_file = 'mnistTrain.tfrecords'
n_samples = x_train.shape[0]
dimension = x_train.shape[1]
with tf.io.TFRecordWriter(record_file) as writer:
   for i in range(n_samples):
      image = x_train[i]
      label = y_train[i]
      tf_example = image_example(image, label, dimension)
      writer.write(tf_example.SerializeToString())



#%% READING FROM TFRECORDS

record_file = 'mnistTrain.tfrecords'

# Create the dataset object from tfrecord file(s)
dataset = tf.data.TFRecordDataset(record_file, buffer_size=100)

# Decoding function
def parse_record(record):
    name_to_features = {
        'dimension': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type=dataType, little_endian=True, fixed_length=None, name=None
    )
    label = record['label']
    dimension = record['dimension']
    image = tf.reshape(image, (dimension, dimension))
    return (image, label)

im_list = []
n_samples_to_show = 16
c = 0
for record in dataset:
  c+=1
  if c > n_samples_to_show:
    break
  parsed_record = parse_record(record)
  decoded_record = decode_record(parsed_record)
  image, label = decoded_record
  im_list.append(image)

# Visualization
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, im_list):
    ax.imshow(im, 'gray')
plt.show()






















