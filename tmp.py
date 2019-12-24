import tensorflow as tf
import numpy as np
import os
import shutil
import random
from PIL import Image
path="D:/yel/detail/512x512/image"
maskpath="D:\yel/detail/512x512/label"

# outpath="D:/tmp"
# file_names=os.listdir(path)  #将该目录下的所有文件名组合成一个list
# file_names=file_names[::]       # 间隔采样
# for file in file_names:     # 循环取文件名
#     img=Image.open(os.path.join(path,file))  # pillow 打开img ，类似opencv imread
#     img=img.resize((1366,768))
#     img.save(os.path.join(outpath,file.split('.')[0]+'.png'))   #保存图片路径

# 删除没有目标的样本
# for file in os.listdir(path):
#     mask=Image.open(os.path.join(maskpath,file))
#     mask=np.array(mask)
#     b=mask.max()
#     if b==0:
#         os.remove(os.path.join(path,file))
#         os.remove(os.path.join(maskpath,file))

# 从样本中随机抽取部分作为val和test,并移动到指定文件
# src_img_path="D:/yel/detail/512x512/image"
# src_path="D:/yel/detail/512x512/label"
# dst_img_path="D:/yel/detail/512x512/test"
# dst_path="D:/yel/detail/512x512/testannot"
# # index=np.random.random_integers(3200,size=(200,))
# files=os.listdir("D:/yel/detail/512x512/image")
# files=random.sample(files,435)
# for i in files:
#     src_img=os.path.join(src_img_path,i)
#     src_mask=os.path.join(src_path,i)
#     dst_img=os.path.join(dst_img_path,i)
#     dst_mask=os.path.join(dst_path,i)
#     shutil.move(src_img,dst_img)
#     shutil.move(src_mask,dst_mask)

#写txt文件
img_path="D:/yel/detail/512x512/val"
label_path="D:/yel/detail/512x512/valannot"
f=open("D:/yel/detail/512x512/val.txt",'w')
for i in os.listdir(img_path):
    f.write(os.path.join(img_path,i)+' '+os.path.join(label_path,i)+'\n')
f.close()