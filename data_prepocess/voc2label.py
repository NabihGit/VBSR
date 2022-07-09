import cv2
import os
import math
import numpy
# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def voc2label_patch(img):
        img_size = img.shape
        img_label = numpy.zeros((img_size[0],img_size[2],img_size[3]))
        for i in range(0,img_size[0]):
            for j in range(0,img_size[2]):
                for k in range(0,img_size[3]):
                    img_label[i,j,k] = ((img[i,0,j,k] & 2**7 ) >> 7 )| ((img[i,0,j,k] & 2**6 ) >> 3 ) | ((img[i,1,j,k] & 2 ** 7) >> 6) | ((img[i,1,j,k] & 2 ** 6) >> 2) | ((img[i,2,j,k] & 2**7 ) >> 5 )
                    if img_label[i,j,k] == 31:
                        img_label[i, j, k] = 21
        return img_label.astype(numpy.uint8)

def voc2label(img):
        img_size = img.shape
        img_label = numpy.zeros((img_size[0],img_size[2]))
        for i in range(0,img_size[0]):
            for j in range(0,img_size[1]):
                img_label[i,j] = ((img[i,j,2] & 2**7 ) >> 7 )| ((img[i,j,2] & 2**6 ) >> 3 ) | ((img[i,j, 1] & 2 ** 7) >> 6) | ((img[i,j, 1] & 2 ** 6) >> 2) | ((img[i,j,0] & 2**7 ) >> 5 )
                if img_label[i,j] == 31:
                    img_label[i, j] = 21
        return img_label.astype(numpy.uint8)

def label2voc(img_label):
        img_size = img_label.shape
        img = numpy.zeros((img_size[0], 3, img_size[1], img_size[2]))
        for i in range(0, img_size[0]):
            for j in range(0, img_size[1]):
                for k in range(0, img_size[2]):
                    temp = img_label[i, j, k]
                    img[i, 2, j, k] = (temp & 2**0) << (7 - 0) | (temp & 2**3) << (6 - 3)
                    img[i, 1, j, k] = (temp & 2 ** 1) << (7 - 1) | (temp & 2 ** 4) << (6 - 4)
                    img[i, 0, j, k] = (temp & 2**2) << (7 - 2)
                    if temp == 21:
                        img[i, 2, j, k] = 224
                        img[i, 1, j, k] = 224
                        img[i, 0, j, k] = 192
        return img

if __name__ == '__main__':
    directory_srcname = "L://Code//pytorch-fcn-main//data//datasets//VOC//VOCdevkit//VOC2012//SegmentationClass//2007_000042.png"
    img = cv2.imread(directory_srcname)
    out_label = voc2label(img)
    out_img = label2voc(out_label)

