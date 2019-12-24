import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
import math

# a=np.array([1,2,3,4])
# b=a[:-1]
#
# cv2.resize()
#
# # 计算所有mask占据图像的比例
imagepath="D:/yel/detail/512x512/train"
labelpath="D:/yel/detail/512x512/trainannot"
mask_number=np.zeros([0])
background_number=[]
for file in os.listdir(labelpath):
    mask=Image.open(os.path.join(labelpath,file))
    mask=np.array(mask).flatten()
    cracks=np.sum(mask)
    backgrounds=mask.shape[0]
    rate=cracks/backgrounds
    mask_number=np.append(mask_number,rate)
mean=np.mean(mask_number)
alpha_1= 0.5*(1+math.exp(-mean))
alpha_2=1-alpha_1
a=0.2595
b=a/mean
print(mean)




# 融合标签和图像
# valuepath="D:/yel/goaf/test_image/densev7_20181228"
# imgpath="D:/yel/refine/512x512/test"
# outpath="D:/yel\goaf/image_fusion/densev7_20181228"
# for file in os.listdir(valuepath):
#     img=cv2.imread(os.path.join(imgpath,file))
#     mask=cv2.imread(os.path.join(valuepath,file),0)
#     mask=cv2.resize(mask,(512,512))
#     print(mask.shape)
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             if mask[i][j]==38:
#                 img[i][j][2]=200
#     cv2.imwrite(os.path.join(outpath,file),img)




# # our NN's output
# logits = tf.constant([[1.0, 2.0, 3.0,4.0], [1.0, 2.0, 3.0,4.0], [1.0, 2.0, 3.0,4.0]])
# # step1:do softmax
# y = tf.nn.softmax(logits)
# # true label
# y_ = tf.constant([[0.0, 0.0, 1.0,0.0], [0.0, 0.0, 1.0,0.0], [0.0, 0.0, 1.0,0.0]])
# # step2:do cross_entropy
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y),axis=1)
# # do cross_entropy just one step
# cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y_))  # dont forget tf.reduce_sum()!!
#
# with tf.Session() as sess:
#     softmax = sess.run(y)
#     c_e = sess.run(cross_entropy)
#     c_e2 = sess.run(cross_entropy2)
#     print("step1:softmax result=")
#     print(softmax)
#     print("step2:cross_entropy result=")
#     print(c_e)
#     print("Function(softmax_cross_entropy_with_logits) result=")
#     print(c_e2)
# mask=Image.open('D:\AerialGoaf/resize/valannot/0000257.png')
# mask=np.array(mask)
# plt.imshow(mask)
# plt.show()
# cv2.waitKey()
# path="D:/zmhj_photo/20181022"
# despath="D:/zmhj_photo/paranoma"

# 文件复制
# files=os.listdir(path)
# files=files[::3]
# for i in files:
#     if 'overlap' in i:
#         continue
#     shutil.copy(os.path.join(path,i),despath)
# for i in os.listdir(despath):
#     image=cv2.imread(os.path.join(despath,i))
#     cv2.imwrite(os.path.join(despath,i),image)


# testpath="D:\AerialGoaf/refine/val"
# testannotpath="D:\AerialGoaf/refine/valmask"
# outpath="D:\AerialGoaf/refine/512x512/val"
# outannotpath="D:\AerialGoaf/refine/512x512/valannot"
# crop_size=512
# # 对图片进行随机裁剪，裁剪出512X512的图片
# for i in os.listdir(testpath):
#     image=Image.open(os.path.join(testpath,i))
#     width,height=image.size
#     mask=Image.open(os.path.join(testannotpath,i))
#     for j in range(5):
#         xa1=random.randint(0,width-crop_size)
#         ya1=random.randint(0,height-crop_size)
#         xa2=xa1+crop_size
#         ya2=ya1+crop_size
#         rect_image=image.crop([xa1,ya1,xa2,ya2])
#         rect_mask=mask.crop([xa1,ya1,xa2,ya2])
#         if not os.path.exists(outpath):
#             os.mkdir(outpath)
#         rect_image.save(os.path.join(outpath,i.split('.')[0]+'_{}.'.format(j)+'png'))
#         if not os.path.exists(outannotpath):
#             os.mkdir(outannotpath)
#         rect_mask.save(os.path.join(outannotpath,i.split('.')[0]+'_{}.'.format(j)+'png'))

# for i in os.listdir(outpath):
#     image=Image.open(os.path.join(outpath,i))
#     mask=Image.open(os.path.join(outannotpath,i))
#     mask=np.array(mask)
#     plt.subplot(121)
#     plt.imshow(image)
#     plt.subplot(122)
#     plt.imshow(mask)
#     plt.show()