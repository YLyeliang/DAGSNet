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
from math import ceil
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)  # D:/tmp\segnet/testing
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)  # D:/tmp\segnet/finetune
tf.app.flags.DEFINE_integer('batch_size', "3", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "D:/yel/goaf/log2", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "D:/yel/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "D:/yel/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "D:/yel/CamVid/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "50000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "360", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "480", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "11", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)


MAX_STEP = 5000
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_CHANNEL = 3
BATCH_SIZE = 5
NUM_CLASSES = 12

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.


# dataset functionality
def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
        im = Image.open(im_filename)
        im = np.array(im, np.float32)
        im = im[np.newaxis]
        la = Image.open(la_filename)
        la = np.array(la)
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
    label_bytes = tf.image.decode_png(labelValue)

    image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    # image = tf.image.resize_images(image, (256,256))
    # label = tf.image.resize_images(label, (256,256))
    # image = tf.image.random_brightness(image,max_delta=63,seed=1)
    # image = tf.image.random_contrast(image,lower=0.2,upper=1.8,seed=1)
    # image = tf.image.random_flip_left_right(image,seed=1)
    # image = tf.image.random_flip_up_down(image,seed=1)
    # label = tf.image.random_flip_left_right(label,seed=1)
    # label = tf.image.random_flip_up_down(label,seed=1)

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

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


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


def storeImageQueue(data, labels, step):
    """ data and labels are all numpy arrays """
    for i in range(BATCH_SIZE):
        index = 0
        im = data[i]
        la = labels[i]
        im = Image.fromarray(np.uint8(im))
        im.save("batch_im_s%d_%d.png" % (step, i))
        writeImage(np.reshape(la, (360, 480)), "batch_la_s%d_%d.png" % (step, i))


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
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        # print("    class # %d accuracy = %f " % (ii, acc))
        print("     class # %d Iou = %f" %(ii,iu[ii]))


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def loss(logits, labels):
    """
        loss func without re-weighting
    """
    # Calculate the average cross entropy loss across the batch.
    logits = tf.reshape(logits, (-1, NUM_CLASSES))
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


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

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def cal_loss(logits, labels):
    loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
      1.0974]) # class 0~11

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def separable_conv_with_bn(inputT, shape, train_phase, activation=True, name=None):
    with tf.variable_scope(name) as scope:
        depth_filter = _variable_with_weight_decay('depth_weights', shape=shape[:-1], initializer=orthogonal_initializer(), wd=None)
        point_filter = _variable_with_weight_decay('point_weights', shape=[1,1,shape[2]*shape[3],shape[4]], initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.separable_conv2d(inputT,depthwise_filter=depth_filter,pointwise_filter=point_filter,strides=[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases', [shape[4]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def atrous_conv2d(inputT,shape,train_phase,rate=2,activation=True,name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.atrous_conv2d(inputT, kernel,rate=rate,padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv

def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope + "_bn",
                                                        reuse=True))

def conv_concate(x, concate_layer,conv_size, growth_rate,phase_train, name):
    shape = x.get_shape().as_list()
    if not concate_layer:
        l = conv_layer_with_bn(x, [conv_size[0], conv_size[1], shape[3], growth_rate], phase_train, name=name)
    else:
        l = tf.concat(concate_layer, axis=3)
        l = tf.concat([x, l], axis=3)
        l = conv_layer_with_bn(l, [conv_size[0], conv_size[1], l.get_shape().as_list()[3], growth_rate], phase_train, name=name)
    return l


def dense_block(l,layers=4,growth_rate=12,conv_size=(3,3),phase_train=True):
    a = []
    a.append(l)
    for i in range(layers):
        l = conv_concate(l, concate_layer=a[:i],conv_size=conv_size, growth_rate=growth_rate,phase_train=phase_train, name='conv_{}.'.format(i))
        if not i == layers - 1:
            a.append(l)
        else:
            a = tf.concat(a, axis=3)
            l = tf.concat([a, l], axis=3)
    return l

def crack_refine(l,conv_size=(7,7),phase_train=True):
    # cr = conv_layer_with_bn(l,[1,7,l.get_shape().as_list()[3],64],phase_train,False,name='CR_0')
    # cr = conv_layer_with_bn(cr,[7,1,cr.get_shape().as_list()[3],64],phase_train,False,name='CR_1')
    cr = separable_conv_with_bn(l,[conv_size[0],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_0')
    cr = separable_conv_with_bn(cr,[1,conv_size[1],cr.get_shape().as_list()[3],2,32],phase_train,True,name='CR_1')
    L = tf.nn.dropout(cr,0.6)
    return L

def crack_refine_new(l,conv_size=(3,3,7,7,11,11),phase_train=True):
    branch_1 = separable_conv_with_bn(l,[conv_size[0],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_00')
    branch_1 = separable_conv_with_bn(branch_1,[1,conv_size[1],branch_1.get_shape().as_list()[3],2,32],phase_train,True,name='CR_01')

    branch_2 = separable_conv_with_bn(l,[conv_size[2],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_10')
    branch_2 = separable_conv_with_bn(branch_2,[1,conv_size[3],branch_2.get_shape().as_list()[3],2,32],phase_train,True,name='CR_11')

    branch_3 = separable_conv_with_bn(l,[conv_size[4],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_20')
    branch_3 = separable_conv_with_bn(branch_3, [1, conv_size[5], branch_2.get_shape().as_list()[3], 2, 32],
                                      phase_train, True, name='CR_11')
    max = tf.math.maximum(branch_1,branch_2)
    L = tf.math.maximum(max,branch_3)
    return L

def atrous_SPP(l,phase_train=True):
    L1 = conv_layer_with_bn(l,[3,3,l.get_shape().as_list()[3],64],phase_train,name='conv_1x1')
    L2 = atrous_conv2d(l,[3,3,l.get_shape().as_list()[3],64], rate=3,train_phase=phase_train, name='atrous_6')
    L3 = atrous_conv2d(l, [3,3,l.get_shape().as_list()[3],64], rate=6,train_phase=phase_train, name='atrous_12')
    L4 = atrous_conv2d(l, [3,3,l.get_shape().as_list()[3],64], rate=9,train_phase=phase_train, name='atrous_18')
    L = tf.concat([L1, L2, L3, L4], axis=3)
    return L

def inference(images, labels, batch_size, phase_train):
    # norm1
    # norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,  # local response normalization.
    #                   name='norm1')
    norm1 = batch_norm_layer(images,phase_train,scope='norm1')
    norm2 = tf.image.resize_bilinear(norm1,(180,240))
    norm3 = tf.image.resize_bilinear(norm1,(90,120))
    norm4 = tf.image.resize_bilinear(norm1,(45,60))
    # input_dense_block_1
    with tf.variable_scope('input_block_1'):
        conv1 = dense_block(norm1,layers=4,growth_rate=16,phase_train=phase_train) #256x256
        conv1 = conv_layer_with_bn(conv1,[1,1,conv1.get_shape().as_list()[3],64],phase_train,True,name='transition')     # trainsition
    # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1') #128x128
    # input_dense_block_2
    with tf.variable_scope('input_block_2'):
        conv2 = dense_block(norm2,layers=4,growth_rate=16,phase_train=phase_train)        #128x128
        conv2 = conv_layer_with_bn(conv2, [1, 1, conv2.get_shape().as_list()[3], 64], phase_train, True,
                                   name='transition')  # trainsition

    # dense_block_1
    with tf.variable_scope('dense_block_1'):
        dense1 = tf.concat([pool1,conv2],axis=3)
        dense1 = dense_block(dense1,layers=4,growth_rate=20,phase_train=phase_train)
        dense1 = conv_layer_with_bn(dense1,[1,1,dense1.get_shape().as_list()[3],64],phase_train,True,name='transition')
    # pool2
        pool2 = tf.nn.max_pool(dense1, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2') #64x64
    # input_dense_block_3
    with tf.variable_scope('input_block_3'):
        conv3 = dense_block(norm3,layers=4,growth_rate=16,conv_size=(3,3),phase_train=phase_train)        #64x64
        conv3 = conv_layer_with_bn(conv3, [1, 1, conv3.get_shape().as_list()[3], 64], phase_train, True,
                                   name='transition')  # trainsition
    # dense_block_2
    with tf.variable_scope('dense_block_2'):
        dense2 = tf.concat([pool2, conv3], axis=3)
        dense2 = dense_block(dense2, layers=4, growth_rate=20, phase_train=phase_train)
        dense2 = conv_layer_with_bn(dense2, [1, 1, dense2.get_shape().as_list()[3], 64], phase_train, True,
                                    name='transition')
    # pool3
        pool3 = tf.nn.max_pool(dense2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # input_dense_block_4
    with tf.variable_scope('input_block_4'):
        conv4 = dense_block(norm4,layers=4,growth_rate=16,conv_size=(3,3),phase_train=phase_train)          #32x32
        conv4 = conv_layer_with_bn(conv4, [1, 1, conv4.get_shape().as_list()[3], 64], phase_train, True,
                                   name='transition')  # trainsition
    # dense_block_3
    with tf.variable_scope('dense_block_3'):
        dense3 = tf.concat([pool3, conv4], axis=3)
        dense3 = dense_block(dense3, layers=4, growth_rate=20, phase_train=phase_train)
        dense3 = conv_layer_with_bn(dense3, [1, 1, dense3.get_shape().as_list()[3], 64], phase_train, True,
                                    name='transition')
    # pool4
        pool4 = tf.nn.max_pool(dense3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')   #16x16
    with tf.variable_scope('atrous_SPP'):
        pool4 = atrous_SPP(pool4,phase_train)
        pool4 = conv_layer_with_bn(pool4,[1,1,pool4.get_shape().as_list()[3],256],phase_train,True,name='trainsition')

    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [3, 3, 256, 256], [batch_size, 45, 60, 256], 2, "up4")
    # decode 4
    with tf.variable_scope('decode_4'):
        cr4 = crack_refine(dense3,(3,3),phase_train=phase_train)
        concat4 = tf.concat([upsample4,cr4],axis=3)
        conv_decode4 = conv_layer_with_bn(concat4, [3, 3, concat4.get_shape().as_list()[3], 128], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3 = deconv_layer(conv_decode4, [3, 3, 128, 128], [batch_size, 90, 120, 128], 2, "up3")
    # decode 3
    with tf.variable_scope('decode_3'):
        cr3 = crack_refine(dense2,(5,5),phase_train=phase_train)
        concat3 = tf.concat([upsample3, cr3], axis=3)
        conv_decode3 = conv_layer_with_bn(concat3, [3, 3, concat3.get_shape().as_list()[3], 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2 = deconv_layer(conv_decode3, [3, 3, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
    # decode 2
    with tf.variable_scope('decode_2'):
        cr2 = crack_refine(dense1,(9,9),phase_train=phase_train)
        concat2 = tf.concat([upsample2, cr2], axis=3)
        conv_decode2 = conv_layer_with_bn(concat2, [3, 3, concat2.get_shape().as_list()[3], 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1 = deconv_layer(conv_decode2, [3, 3, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    with tf.variable_scope('decode_1'):
        cr1= crack_refine(conv1,(13,13),phase_train=phase_train)
        concat1 = tf.concat([upsample1, cr1], axis=3)
        conv_decode1 = conv_layer_with_bn(concat1, [3, 3, concat1.get_shape().as_list()[3], 64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0005)
        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    loss = cal_loss(conv_classifier, labels)

    return loss \
        , logit


def train(total_loss, global_step):
    total_sample = 274
    num_batches_per_epoch = 274 / 1
    """ fix lr """
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps=5000, decay_rate=0.95)
    # lr = INITIAL_LEARNING_RATE
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


def test(FLAGS):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir  # /tmp3/first350/TensorFlow/Logs
    test_dir = FLAGS.test_dir  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    test_ckpt = FLAGS.testing
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    # testing should set BATCH_SIZE = 1
    batch_size = 1

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

    test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h,image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, logits = inference(test_data_node, test_labels_node, batch_size, phase_train)

    pred = tf.argmax(logits, axis=3)
    # get moving avg
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Load checkpoint
        ckpt = "D:/yel/goaf/log2/model.ckpt-43000"
        if ckpt:
            saver.restore(sess, ckpt)

        images, labels = get_all_test_data(image_filenames, label_filenames)

        threads = tf.train.start_queue_runners(sess=sess)
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        count =0
        for image_batch, label_batch in zip(images, labels):

            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                phase_train: False
            }

            dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
            # output_image to verify
            # if (FLAGS.save_image):
                # writeImage(im[0], 'testing_image.png')
                # writeImage(im[0], 'D:/tmp\crack/val_7_256_dense_concat/'+str(image_filenames[count]).split('\\')[-1])

            hist += get_hist(dense_prediction, label_batch)
            count+=1
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("acc: ", acc_total)
        print("mean IU: ", np.nanmean(iu))
        for ii in range(NUM_CLASSES):
            if float(hist.sum(1)[ii]) == 0:
                acc = 0.0
            else:
                acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
            print("    class # %d accuracy = %f " % (ii, acc))
            print("     class # %d Iou = %f" % (ii, iu[ii]))



def training(FLAGS, is_finetune=False):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir
    image_dir = FLAGS.image_dir
    val_dir = FLAGS.val_dir
    finetune_ckpt = FLAGS.finetune
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    # should be changed if your model stored by different convention
    startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

    image_filenames, label_filenames = get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)

    with tf.Graph().as_default():

        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])

        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        global_step = tf.Variable(0, trainable=False)

        # For CamVid
        images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)

        val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size)

        # Build a Graph that computes the logits predictions from the inference model.
        loss, eval_prediction = inference(train_data_node, train_labels_node, batch_size, phase_train)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=12)

        summary_op = tf.summary.merge_all()

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Build an initialization operation to run below.
            if (is_finetune == True):
                saver.restore(sess, finetune_ckpt)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            # Start the queue runners.
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

                    # eval current training batch pre-class accuracy
                    pred = sess.run(eval_prediction, feed_dict=feed_dict)
                    per_class_acc(pred, label_batch)

                if step % 100 == 0:
                    print("start validating.....")
                    total_val_loss = 0.0
                    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
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
                    print(" end validating.... ")

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

test(FLAGS)