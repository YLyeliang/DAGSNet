import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from PIL import Image

# img=cv2.imread("D:/AerialGoaf/refine/train/0009.png")
# img=cv2.rectangle(img,(300,300),(556,556),color=(0,128,128),thickness=5)
# cv2.imshow("ha",img)
# cv2.waitKey()

with tf.device('/gpu:1'):


# IMAGE_HEIGHT=512
# IMAGE_WIDTH=512
# IMAGE_CHANNEL=3
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=200
# image_dir="D:/AerialGoaf/refine/512x512/train.txt"
#
# def get_filename_list(path):
#     fd = open(path)
#     image_filenames = []
#     label_filenames = []
#     for i in fd:
#         i = i.strip().split(" ")
#         image_filenames.append(i[0])
#         label_filenames.append(i[1])
#     return image_filenames, label_filenames
#
#
# def CamVid_reader(filename_queue):
#     image_filename = filename_queue[0]
#     label_filename = filename_queue[1]
#
#     imageValue = tf.read_file(image_filename)
#     labelValue = tf.read_file(label_filename)
#
#     image_bytes = tf.image.decode_png(imageValue)
#     label_bytes = tf.image.decode_png(labelValue, dtype=tf.uint16)
#
#     image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
#     label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
#     image = tf.image.crop_and_resize()
#     image = tf.image.resize_images(image, (256,256))
#     label = tf.image.resize_images(label, (256,256))
#     image = tf.image.random_brightness(image,max_delta=30,seed=1)
#     image = tf.image.random_contrast(image,lower=0.2,upper=1.8,seed=1)
#     image = tf.image.random_flip_left_right(image,seed=1)
#     image = tf.image.random_flip_up_down(image,seed=1)
#     label = tf.image.random_flip_left_right(label,seed=1)
#     label = tf.image.random_flip_up_down(label,seed=1)
#
#     return image, label
#
# def _generate_image_and_label_batch(image, label, min_queue_examples,
#                                     batch_size, shuffle):
#     """Construct a queued batch of images and labels.
#
#     Args:
#       image: 3-D Tensor of [height, width, 3] of type.float32.
#       label: 3-D Tensor of [height, width, 1] type.int32
#       min_queue_examples: int32, minimum number of samples to retain
#         in the queue that provides of batches of examples.
#       batch_size: Number of images per batch.
#       shuffle: boolean indicating whether to use a shuffling queue.
#
#     Returns:
#       images: Images. 4D tensor of [batch_size, height, width, 3] size.
#       labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
#     """
#     # Create a queue that shuffles the examples, and then
#     # read 'batch_size' images + labels from the example queue.
#     num_preprocess_threads = 1
#     if shuffle:
#         images, label_batch = tf.train.shuffle_batch(
#             [image, label],
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             capacity=min_queue_examples + 3 * batch_size,
#             min_after_dequeue=min_queue_examples)
#     else:
#         images, label_batch = tf.train.batch(
#             [image, label],
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             capacity=min_queue_examples + 3 * batch_size)
#
#     # Display the training images in the visualizer.
#     # tf.image_summary('images', images)
#
#     return images, label_batch
#
#
# def CamVidInputs(image_filenames, label_filenames,batch_size):
#     images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
#     labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)
#
#     filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
#
#     image, label = CamVid_reader(filename_queue)
#     reshaped_image = tf.cast(image, tf.uint8)
#
#     min_fraction_of_examples_in_queue = 0.4
#     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
#                              min_fraction_of_examples_in_queue)
#     print('Filling queue with %d CamVid images before starting to train. '
#           'This will take a few minutes.' % min_queue_examples)
#
#     # Generate a batch of images and labels by building up a queue of examples.
#     return _generate_image_and_label_batch(reshaped_image, label,
#                                            min_queue_examples, batch_size,
#                                            shuffle=True)
#
#
# image_filenames,label_filenames=get_filename_list(image_dir)
# images, labels = CamVidInputs(image_filenames, label_filenames,3)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in range(1000):
#         img,la = sess.run([images,labels])
#         cv2.imshow("1",img[0])
#         cv2.imshow("2",img[1])
#         cv2.imshow("3",img[2])
#         cv2.imshow("la1",la[0])
#         cv2.waitKey()
#     coord.request_stop()
#     coord.join(threads)
