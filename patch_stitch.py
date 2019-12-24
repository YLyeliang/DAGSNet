import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
# from multi_input_with_max import inference
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def image_crop(image,out_path,img_name):
    images=[]
    image=np.array(image,np.float32)
    for i in range(6):
        if i<2:
            im = image[:512,i*455:512+i*455]
        elif i==2:
            im = image[:512,854:]
        elif i>2 and i <5:
            im = image[256:,(i-3)*455:512+(i-3)*455]
        else:
            im = image[256:,854:]
        im = Image.fromarray(np.uint8(im)).resize((512,512))
        im.save(out_path+'/'+img_name.split(".")[0]+'_{}'.format(i)+".png")
        images.append(im)
    return images

path="D:/yel/detail/resource/label"
out="D:/yel/detail/resource/patchannot"
for i in os.listdir(path):
    img=Image.open(os.path.join(path,i))
    images=image_crop(img,out,i)

# def predict():
#     batch_size=1
#     image_h=512
#     image_w=512
#     image_c=3
#
#     test_data_node = tf.placeholder(tf.float32,
#                                     shape=[batch_size, image_h, image_w, image_c])
#
#     test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h,image_w, 1])
#
#     phase_train = tf.placeholder(tf.bool, name='phase_train')
#
#     loss, logits = inference(test_data_node,test_labels_node, batch_size, phase_train)
#
#     pred = tf.argmax(logits, axis=3)
#     # get moving avg
#     variable_averages = tf.train.ExponentialMovingAverage(
#         0.999)
#     variables_to_restore = variable_averages.variables_to_restore()
#
#     saver = tf.train.Saver(variables_to_restore)
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     with tf.Session(config=config) as sess:
#         # Load checkpoint
#         ckpt = "D:/yel/goaf/lab/multi_input_0.005_crmax/model.ckpt-35000"
#         if ckpt:
#             saver.restore(sess, ckpt)
#
#         # images, labels = get_all_test_data(image_filenames, label_filenames)
#
#         threads = tf.train.start_queue_runners(sess=sess)
#         count = 0
#         for image_batch in images:
#             feed_dict = {
#                 test_data_node: image_batch,
#                 test_labels_node:None,
#                 phase_train: False
#             }
#             dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
#             # output_image to verify
#             # writeImage(im[0], '028_5.png')
#             # writeImage(im[0], 'D:/yel/goaf/test_image/onepart_training/'+str(image_filenames[count]).split('\\')[-1])