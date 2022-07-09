import cv2
import os
import math

# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_srcname,directory_desname,SamplePath_src,SamplePath_des):
    for filename in os.listdir(directory_srcname):
        portion = os.path.splitext(filename)
        #print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_srcname + "/" + filename)
        img2 = cv2.imread(SamplePath_src + "/" + portion[0] + ".jpg")
        #####显示图片#######
        #cv2.imshow(filename, img2)
        #cv2.waitKey(0)
        #####################
        img_size = img.shape
        top_size, bottom_size, left_size, right_size = (math.ceil((500-img_size[0])/2), math.floor((500-img_size[0])/2),
                                                        math.ceil((500-img_size[1])/2), math.floor((500-img_size[1])/2))
        const_img  = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
        const_img2 = cv2.copyMakeBorder(img2, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
        #####保存图片#########
        cv2.imwrite(directory_desname + "/" + filename, const_img)
        cv2.imwrite(SamplePath_des + "/" + filename, const_img2)

if __name__ == '__main__':
    directory_srcname = "L:\Code\pytorch-fcn-main\data\datasets\VOC\VOCdevkit\VOC2012\SegmentationClass"
    directory_desname = "L:\Code\pytorch-fcn-main\data\datasets\VOC\VOCdevkit\VOC2012\Train_seg"
    SamplePath_src    = "L:\Code\pytorch-fcn-main\data\datasets\VOC\VOCdevkit\VOC2012\JPEGImages"
    SamplePath_des    =  "L:\Code\pytorch-fcn-main\data\datasets\VOC\VOCdevkit\VOC2012\Train"
    read_directory(directory_srcname, directory_desname, SamplePath_src, SamplePath_des)

