from PIL import Image
import numpy as np
import os


imagepath="D:\yel/detail/512x512/train"
labelpath="D:\yel/detail/512x512/trainannot"
image_out="D:\yel/refine/512x512/train"
label_out="D:\yel/refine/512x512/trainannot"
keep_prob=0.01
mask_number=np.zeros([0])
background_number=[]
count=0
for file in os.listdir(labelpath):
    mask=Image.open(os.path.join(labelpath,file))
    mask=np.array(mask).flatten()
    cracks=np.sum(mask)             # 裂缝像素数
    backgrounds=mask.shape[0]       # 背景像素数
    rate=cracks/backgrounds         # 裂缝占比
    # if rate< keep_prob:
    #     os.remove(os.path.join(labelpath,file))
    #     os.remove(os.path.join(imagepath,file))
    #     count+=1
    mask_number=np.append(mask_number,rate)
mean=np.mean(mask_number)
a=0.2595
b=a/mean
print(mean)
print(count)


