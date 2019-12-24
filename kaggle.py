import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import skimage
import skimage.io
from PIL import Image

FLAGS = tf.app.flags.FLAGS

LOG_DIR=
TRAIN_DIR=
VAL_DIR=
TEST_DIR=
TEST_CKPT=

MAX_STEP = 5000
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 3
BATCH_SIZE = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')


def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array(
        [Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist,
         Unlabelled])
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
        im = np.array(skimage.io.imread(im_filename), np.float32)
        im = im[np.newaxis]
        la = skimage.io.imread(la_filename)
        la = la[np.newaxis]
        la = la[..., np.newaxis]
        images.append(im)
        labels.append(la)
    return images, labels


def get_filename_list(path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return image_filenames, label_filenames


def CamVid_reader(filename_queue):
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imageValue = tf.read_file(image_filename)
    labelValue = tf.read_file(label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue, dtype=tf.uint16)

    image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 3-D Tensor of [height, width, 1] type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.image_summary('images', images)

    return images, label_batch


def CamVidInputs(image_filenames, label_filenames, batch_size):
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    image, label = CamVid_reader(filename_queue)
    reshaped_image = tf.cast(image, tf.float32)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CamVid images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def cal_loss(logits, labels):
    loss_weight = np.array([
        0.2595,
        9.0974])  # class 0~11

    labels = tf.cast(labels, tf.int32)
    return weighted_loss(logits, labels, num_classes=2, head=loss_weight)


def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  # 带权重的softmax 交叉熵loss
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual los.moving average ses and the total loss; do the
    # same for the averaged version of the lossesversion of the loss
    for l in losses + [total_loss]:
        # as the original loss name.
        # Name each loss as '(raw)' and name the
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(predictions, labels):
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def print_hist_summery(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  #iou
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


def batch_norm(x_tensor, name=None):
    mean, variance = tf.nn.moments(x_tensor, axes=[0])
    L = tf.nn.batch_normalization(x_tensor, mean, variance, 0.01, 1, 0.001, name=name)
    return L


def relu(L):
    return tf.nn.relu(L)


def avgpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.avg_pool(x_tensor, ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                          strides=[1, pool_strides[0], pool_strides[1], 1], padding='VALID')


def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name):
    """
    Apply BN-RELU-conv2d to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param conv_ksize: kernel size 2-D Tuple for pool
    :param name: variable scope name
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                                  initializer=orthogonal_initializer(),
                                  regularizer=regularizer)
        biase = tf.get_variable('biase',shape=conv_num_outputs,initializer=tf.constant_initializer(0.0))
        L = batch_norm(x_tensor, 'bn')
        L = relu(L)
        L = tf.nn.conv2d(L, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
        L = tf.nn.bias_add(L,biase)

    return L


def atrous_conv2d(x_tensor, conv_num_outputs, conv_ksize, rate, name):
    """
    Apply BN-RELU-atrous_conv2d to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: Stride 2-D Tuple for atrous convolution
    :param rate: atrous convolution rate
    :param name: variable scope name
    :return: The result tensor
    """
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('atrous_weights', shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                                  initializer=orthogonal_initializer(),
                                  regularizer=regularizer)
        biase = tf.get_variable('biases', shape=conv_num_outputs, initializer=tf.constant_initializer(0.0))
        L = batch_norm(x_tensor, 'bn')
        L = relu(L)
        L = tf.nn.atrous_conv2d(L, weights, rate=rate, padding='SAME')
        L = tf.nn.bias_add(L,biase)
    return L


def atrous_conv2d_nobn(x_tensor, conv_num_outputs, conv_ksize, rate, name):
    """
    Apply atrous-conv2d to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: Stride 2-D Tuple for atrous convolution
    :param rate: atrous convolution rate
    :param name: variable scope name
    :return: The result tensor
    """
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                                  initializer=orthogonal_initializer(),
                                  regularizer=regularizer)
        biase = tf.get_variable('biases', shape=conv_num_outputs, initializer=tf.constant_initializer(0.0))
        L = tf.nn.atrous_conv2d(x_tensor, weights, rate=rate, padding='SAME')
        L = tf.nn.bias_add(L,biase)
    return L


def conv2d_nobn(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                                  initializer=orthogonal_initializer(),
                                  regularizer=regularizer)
        biase = tf.get_variable('biases', shape=conv_num_outputs, initializer=tf.constant_initializer(0.0))
        L = tf.nn.conv2d(x_tensor, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
        L=tf.nn.bias_add(L,biase)
    return L


def conv_concate(x, concate_layer, growth_rate, name):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if not concate_layer:
            l = conv2d(x, conv_num_outputs=growth_rate, conv_ksize=(3, 3), conv_strides=(1, 1), name='bn_relu_conv')
        else:
            l = tf.concat(concate_layer, axis=3)
            l = tf.concat([x, l], axis=3)
            l = conv2d(l, conv_num_outputs=growth_rate, conv_ksize=(3, 3), conv_strides=(1, 1), name='bn_relu_conv')
    return l


def atrous_concate(x, concate_layer, growth_rate, dilate_rate, name):
    with tf.variable_scope(name):
        if not concate_layer:
            l = atrous_conv2d(x, conv_num_outputs=growth_rate, conv_ksize=(3, 3), rate=dilate_rate,
                              name='bn-relu-atrous')
        else:
            l = tf.concat(concate_layer, axis=3)
            l = tf.concat([x, l], axis=3)
            l = atrous_conv2d(l, conv_num_outputs=growth_rate, conv_ksize=(3, 3), rate=dilate_rate,
                              name='bn-relu-atrous')
    return l


def dense_block(l, layers=12, growth_rate=24):
    a = []
    a.append(l)
    for i in range(layers):
        l = conv_concate(l, concate_layer=a[:i], growth_rate=growth_rate, name='conv_{}.'.format(i))
        if not i == layers - 1:
            a.append(l)
        else:
            a = tf.concat(a, axis=3)
            l = tf.concat([a, l], axis=3)
    return l


def dense_atrous_block(l, layers=6, growth_rate=12, dilate_rate=2):
    a = []
    a.append(l)
    for i in range(layers):
        l = atrous_concate(l, concate_layer=a[:i], growth_rate=growth_rate, dilate_rate=dilate_rate,
                           name='atrous_{}'.format(i))
        if not i == layers - 1:
            a.append(l)
        else:
            a = tf.concat(a, axis=3)
            l = tf.concat([a, l], axis=3)
    return l


def atrous_SPP(l, dilate_rate=(6, 12, 18)):
    l = batch_norm(l, name='bn')
    L1 = conv2d_nobn(l, 256, (1, 1), (1, 1), name='conv_1x1')
    L2 = atrous_conv2d_nobn(l, 256, (3, 3), rate=6, name='atrous_6')
    L3 = atrous_conv2d_nobn(l, 256, (3, 3), rate=12, name='atrous_12')
    L4 = atrous_conv2d_nobn(l, 256, (3, 3), rate=18, name='atrous_18')
    L = tf.concat([L1, L2, L3, L4], axis=3)
    return L


def transition(l, name=None):
    with tf.variable_scope(name):
        l = conv2d(l, 16, (1, 1), (1, 1), name='conv_1x1')
        l = avgpool(l, (2, 2), (2, 2))
    return l

def output(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name):
    """
    Apply BN-RELU-conv2d to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param conv_ksize: kernel size 2-D Tuple for pool
    :param name: variable scope name
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                                  initializer=orthogonal_initializer(),
                                  regularizer=regularizer)
        biase = tf.get_variable('biases', shape=2, initializer=tf.constant_initializer(0.0))
        L = batch_norm(x_tensor, 'bn')
        L = relu(L)
        L = tf.nn.conv2d(L, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
        L = tf.nn.bias_add(L, biase)
    return L


# Architecture of The segmentation
def inference(x, labels, grwoth_rate=12, training=None):
    end_points = {}  # x:512*512
    l = conv2d_nobn(x, 16, (5, 5), (2, 2), 'conv0')  # l:256*256
    # Encoder process
    with tf.variable_scope('dense_block1'):
        l = dense_block(l, layers=8)
        l = transition(l, name='transition')  # l:128*128
    with tf.variable_scope('dense_block2'):
        l = dense_block(l, layers=6)
        end_points['dense_block2'] = l
        l = transition(l, name='transition')  # l:64*64
    with tf.variable_scope('dense_block3'):
        l = dense_block(l, layers=6)
        l = transition(l, name='transition')  # l:32*32
    with tf.variable_scope('dense_block4'):
        l = dense_atrous_block(l, layers=6, dilate_rate=2)
    with tf.variable_scope('aspp'):
        l = atrous_SPP(l)
        l = conv2d(l, 256, (1, 1), (1, 1), name='transition')

    # Decoder process
    with tf.variable_scope('decoder_1'):
        l = tf.image.resize_bilinear(l, (128, 128))
        end_points['dense_block2'] = conv2d(end_points['dense_block2'], 256, (1, 1), (1, 1), name='conv_1x1')
        l = tf.concat([l, end_points['dense_block2']], axis=3)

    with tf.variable_scope('upsample'):
        l = output(l, 2, (3, 3), (1, 1), name='output')
        l = tf.image.resize_bilinear(l, (512, 512))
    loss = cal_loss(l, labels)
    return l, loss


def test():
    max_steps = MAX_STEP
    train_dir = TRAIN_DIR  # /tmp3/first350/TensorFlow/Logs
    test_dir = TEST_DIR  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    test_ckpt = TEST_CKPT
    image_w = IMAGE_WIDTH
    image_h = IMAGE_HEIGHT
    image_c = IMAGE_WIDTH
    # testing should set BATCH_SIZE = 1
    batch_size = 1

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

    test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 512,512, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    logits,loss = inference(test_data_node, test_labels_node, batch_size, phase_train)

    pred = tf.argmax(logits, axis=3)
    # get moving avg
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

    with tf.Session() as sess:
        # Load checkpoint
        ckpt = tf.train.get_checkpoint_state(test_ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        images, labels = get_all_test_data(image_filenames, label_filenames)

        threads = tf.train.start_queue_runners(sess=sess)
        hist = np.zeros((2, 2))
        count=0
        for image_batch, label_batch in zip(images, labels):

            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                phase_train: False
            }

            dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
            # output_image to verify
            if (FLAGS.save_image):
                # writeImage(im[0], 'testing_image.png')
                writeImage(im[0], 'D:/tmp\goaf/test_inference/'+str(image_filenames[count]).split("\\")[-1])

            hist += get_hist(dense_prediction, label_batch)
            count+=1
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("acc: ", acc_total)
        print("mean IU: ", np.nanmean(iu))


def train(total_loss, global_step):
    total_sample = 274
    num_batches_per_epoch = 274 / 1
    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def training(is_finetune=False):
    max_steps = 10000
    batch_size = BATCH_SIZE
    train_dir = LOG_DIR
    image_dir = TRAIN_DIR
    val_dir = VAL_DIR
    finetune_ckpt = FLAGS.finetune
    image_h = IMAGE_HEIGHT
    image_w = IMAGE_WIDTH
    image_c = IMAGE_CHANNEL

    # should be changed if your model stored by different convention
    startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

    image_filenames, label_filenames = get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)

    with tf.Graph().as_default():
        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        global_step = tf.Variable(0, trainable=False)

        # For AerialGoaf
        images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)
        val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size)

        # Build a Graph that computes the logits predictions from the inference model.
        eval_prediction, loss = inference(train_data_node, train_labels_node, batch_size, phase_train)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = train(loss, global_step)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            # Build an initialization operation below
            if (is_finetune == True):
                saver.restore(sess, finetune_ckpt)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            # start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Summary placeholders
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            average_pl = tf.placeholder(tf.float32)
            acc_pl = tf.placeholder(tf.float32)
            iu_pl = tf.placeholder(tf.float32)
            average_summary = tf.summary.scalar("test_average_loss", average_pl)
            acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
            iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

            for step in range(startstep, startstep + max_steps):
                image_batch, label_batch = sess.run([images, labels])
                # since we still use mini-batches in validation, still set bn-layer phase_train = True
                feed_dict = {
                    train_data_node: image_batch,
                    train_labels_node: label_batch,
                    phase_train: True
                }
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                    # eval current training batch accuracy
                    pred = sess.run(eval_prediction, feed_dict=feed_dict)
                    per_class_acc(pred, label_batch)
                if step % 100 == 0:
                    print("start validation...")
                    total_val_loss = 0.0
                    hist = np.zeros((2, 2))
                    for test_step in range(int(TEST_ITER)):
                        val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                        _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                            train_data_node: val_images_batch,
                            train_labels_node: val_labels_batch,
                            phase_train: True
                        })
                        total_val_loss += _val_loss
                        hist += get_hist(_val_pred, val_labels_batch)
                    print("val loss: ", total_val_loss / TEST_ITER)
                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
                    acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    print_hist_summery(hist)
                    print("end validation...")

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(test_summary_str, step)
                    summary_writer.add_summary(acc_summary_str, step)
                    summary_writer.add_summary(iu_summary_str, step)
                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == max_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)


training()
