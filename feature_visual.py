import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
# from densev8 import inference         #第一次  densev8
# from multi_input_dense import inference #第二次 multi_input_dense
# from multi_input_with_max import inference #第三次 multi_input_with_max
# from DenseNet_56 import inference
# from multi_input_with_max_2inputs import inference
from multi_input_with_max_3inputs import inference
import matplotlib.pyplot as plt
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MOVING_AVERAGE_DECAY = 0.9999

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


def get_filename_list(path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return image_filenames, label_filenames


def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
        im = Image.open(im_filename).resize((256,256))
        im = np.array(im, np.float32)
        im = im[np.newaxis]
        la = Image.open(la_filename).resize((256,256))
        la = np.array(la)
        la = la[np.newaxis]
        la = la[..., np.newaxis]
        images.append(im)
        labels.append(la)
    return images, labels


def visualize(image, conv_output, conv_grad, gb_viz,file_save="D:/yel/goaf/heatmap/densev8",file_name="heatmap.png"):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (256, 256), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

    path_Grad=os.path.join(file_save,"Grad_CAM")
    if not os.path.exists(path_Grad):
        os.mkdir(path_Grad)
    im=Image.fromarray(cam_heatmap)
    im=im.resize((512,512))
    im.save(os.path.join(path_Grad,file_name))

    gb_viz = np.dstack((
        gb_viz[:, :, 0],
        gb_viz[:, :, 1],
        gb_viz[:, :, 2],
    ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(132)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')

    path_guidebp=os.path.join(file_save,"guided_BP")
    if not os.path.exists(path_guidebp):
        os.mkdir(path_guidebp)
    # cv2.imwrite(os.path.join(path_guidebp,file_name),gb_viz)
    save_gb_viz=(gb_viz*255).astype(np.uint8)
    im2=Image.fromarray(save_gb_viz)
    im2=im2.resize((512,512))
    im2.save(os.path.join(path_guidebp,file_name))

    gd_gb = np.dstack((
        gb_viz[:, :, 0] * cam,
        gb_viz[:, :, 1] * cam,
        gb_viz[:, :, 2] * cam,
    ))
    ax = fig.add_subplot(133)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    path_guideGrad=os.path.join(file_save,"guided_GradCAM")
    if not os.path.exists(path_guideGrad):
        os.mkdir(path_guideGrad)
    save_gd_gb=(gd_gb*255).astype(np.uint8)
    im3=Image.fromarray(save_gd_gb)
    im3=im3.resize((512,512))
    im3.save(os.path.join(path_guideGrad,file_name))

    # plt.show()



def visual():
    batch_size = 5
    test_dir = "D:/yel/detail/512x512/test.txt"  # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
    image_w = 256
    image_h = 256
    image_c = 3
    # testing should set BATCH_SIZE = 1
    batch_size = 1

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

    test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h,image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, logits,pool3 = inference(test_data_node, test_labels_node, batch_size, phase_train)

    pred = tf.argmax(logits, axis=3)

    # 版本1
    label = tf.one_hot(test_labels_node, depth=2)
    label = tf.squeeze(label,axis=[3])
    y_c = tf.reduce_sum(tf.multiply(logits, label), axis=-1)

    #版本2
    # label_flat = tf.reshape(test_labels_node, (-1, 1))
    # label = tf.reshape(tf.one_hot(label_flat, depth=2), (-1, 2))
    # logits = tf.reshape(logits,(-1,2))
    # y_c = tf.reduce_sum(tf.multiply(logits,label),axis=-1)
    print('y_c:', y_c)
    # Get last convolutional layer gradient for generating gradCAM visualization
    target_conv_layer = pool3
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

    # Guided backpropagtion back to input layer
    gb_grad = tf.gradients(loss, test_data_node)[0]

    # get moving avg
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Load checkpoint
        ckpt = "D:/yel/goaf/lab/multi_input_dense_max_3inputs/model.ckpt-39000"
        if ckpt:
            saver.restore(sess, ckpt)

        images, labels = get_all_test_data(image_filenames, label_filenames)

        threads = tf.train.start_queue_runners(sess=sess)
        hist = np.zeros((2, 2))
        count =0

        for image_batch, label_batch in zip(images, labels):

            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                phase_train: False
            }

            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value,dense_prediction = sess.run(
                [gb_grad, target_conv_layer, target_conv_layer_grad,logits], feed_dict=feed_dict)

            visualize(image_batch[0], target_conv_layer_value[0], target_conv_layer_grad_value[0], gb_grad_value[0],
                      file_save="D:/yel/goaf/heatmap/multi_input_max_3inputs\pool4",file_name=str(image_filenames[count]).split('\\')[-1])
            # output_image to verify
            # if (FLAGS.save_image):
                # writeImage(im[0], 'testing_image.png')
                # writeImage(im[0], 'D:/yel/goaf/test_image/densev7_20181228/'+str(image_filenames[count]).split('\\')[-1])

            hist += get_hist(dense_prediction, label_batch)
            count+=1
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("acc: ", acc_total)
        print("crack accuracy: ", np.diag(hist)[1]/hist.sum(1)[1])
        print("background accuracy: ",np.diag(hist)[0]/hist.sum(1)[0])
        print("mean IU: ", np.nanmean(iu))
        print('background IU = ',iu[0])
        print('crack IU = ',iu[1])

visual()