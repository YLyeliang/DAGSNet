import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

image_path="D:/AerialGoaf/512x512/train"
mask_path="D:/AerialGoaf/512x512/trainannot"

for file in os.listdir(image_path):
    image=cv2.imread(os.path.join(image_path,"00002_1.png"),0)
    mask=Image.open(os.path.join(mask_path,"00002_1.png"))
    mask=np.array(mask)


    hist,bins=np.histogram(image.flatten(),256,[0,256])

    # 直方图均衡化
    cdf=hist.cumsum() #calculate historgram
    cdf_normalized =cdf *hist.max()/cdf.max()
    cdf_m=np.ma.masked_equal(cdf,0)
    cdf_m= (cdf_m -cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())
    # 对被掩盖的元素赋值，这里赋值为 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[image]
    cv2.imshow("origin",image)
    cv2.imshow('p', img2)
    img3=cv2.blur(img2,(3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    ret,thresh=cv2.threshold(img2,40,255,cv2.THRESH_BINARY_INV)
    dilate=cv2.dilate(thresh,kernel)
    dilate=cv2.erode(dilate,kernel)
    ret2,thresh2=cv2.threshold(img3,100,255,cv2.THRESH_BINARY)
    ret3,thresh3=cv2.threshold(img2,0,255,cv2.THRESH_OTSU)
    lbp =local_binary_pattern(image,40,5)
    cv2.imshow("lbp",lbp)
    cv2.imshow("thresh",thresh)
    cv2.imshow("dilate",dilate)
    cv2.imshow("thresh2",thresh2)
    cv2.imshow("thresh3", thresh3)
    plt.imshow(mask)
    plt.show()
    cv2.waitKey()

    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()




