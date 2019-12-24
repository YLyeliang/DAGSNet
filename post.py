import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt


valuepath="D:/tmp\crack/val_7_256_dense_separable_block"
imgpath="D:\AerialGoaf/512x512/test"
outpath="D:/tmp\crack/val_7_fusion"
# 将Mask标注于原图上
for file in os.listdir(valuepath):
    img=cv2.imread(os.path.join(imgpath,file))
    mask=cv2.imread(os.path.join(valuepath,file),0)
    mask=cv2.resize(mask,(512,512))
    print(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]==38:
                img[i][j][2]=255
    cv2.imwrite(os.path.join(outpath,file),img)

# for file in os.listdir(valuepath):
#     mask=cv2.imread(os.path.join(valuepath,file))
#     mask=cv2.resize(mask,(512,512))
#     cv2.imwrite(os.path.join(outpath,file),mask)
