import cv2
import math
import numpy as np

# 计算图像的暗通道

def getdark(im,sz):
    im = np.array(im)
    b,g,r = np.split(im,1,axis=2)
    c=1



def DarkChannel(im, sz):
    b, g, r = cv2.split(im)         #取 R,G,B通道
    dc = cv2.min(cv2.min(r, g), b)          #取三通道的最小值作为暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)    #腐蚀暗通道
    # dark = dc
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]       #dark为RGB中最小的值
    imsz = h * w                #获取宽高
    numpx = int(max(math.floor(imsz / 1000), 1))    # 像素规模/1000？
    darkvec = dark.reshape(imsz, 1)     #暗通道    flatten
    imvec = im.reshape(imsz, 3)         #RGB
    indices = darkvec.argsort(axis=0)     #获取数组从小到大的值索引
    indices = indices[imsz - numpx::]   #获取第imsz-numpx之后的所有值

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


if __name__ == '__main__':
    import sys
    import os
    fn_path = "D:/yel/detail/resource/image"
    out_path= "d:/yel/detail/resource/Haze_removal"
    for i in os.listdir(fn_path):
        fn=os.path.join(fn_path,i)
        # fn= "haha.jpg"
        src = cv2.imread(fn,cv2.IMREAD_COLOR)
        I = src.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(src, te)
        J = Recover(I, t, A, 0.1)
        arr = np.hstack((I, J))
        cv2.imshow("contrast", arr)
        # dark_show=dark*255
        # dark_show=dark_show.astype('uint8')
        # cv2.imshow("darkChannel",dark_show)
        # a,dark_mask=cv2.threshold(dark_show,50,255,cv2.THRESH_BINARY)
        # dark_mask = cv2.adaptiveThreshold(dark_show, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        # cv2.imshow("darkMask",dark_mask)
        cv2.imwrite(os.path.join(out_path,i), J * 255)
        # src_mask=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
        # cv2.imshow("src-gray",src_mask)
        # mask=cv2.imread("dehaze.png",0)
        # cv2.imshow("gray",mask)
        # diff_mask=src_mask-mask
        # diff_mask=cv2.max(diff_mask,0)
        # _, diff_mask2 = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
        # cv2.imshow("diff",diff_mask)
        # cv2.imshow("diffMask",diff_mask2)
        # mask2=cv2.adaptiveThreshold(src_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # ret,mask = cv2.threshold(mask,50,255,cv2.THRESH_BINARY)
        # mask=cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # mask = cv2.dilate(mask, kernel)
        # mask = cv2.erode(mask, kernel)
        # cv2.imshow("src_mask",mask2)
        # cv2.imshow("mask",mask)
        # cv2.imwrite("contrast.png", arr * 255)
        # cv2.waitKey()
