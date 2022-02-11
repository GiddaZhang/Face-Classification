import time
import numpy as np
import cv2
from os import name
from PIL import Image
from tqdm import tqdm, trange

dataPath = 'Dataset/'
newPath = 'Dataset/'+'Image-haired/'
count_male = 0
count_female = 0

file_male = open(r'Dataset/Label/male_names.txt', 'r')
maleNameList = list(file_male)  # 创建男性姓名列表
for i in range(len(maleNameList)):
    maleNameList[i] = maleNameList[i].strip('\n')  # 读的人名带\n，去掉之
print('Detecting male faces...')

for i in tqdm(maleNameList):
    name = i[:i.rfind("_")]
    picPath = dataPath + 'Image' + \
        '/'+str(name)+'/'+str(i)

    # 以下进行图像的读入与裁剪
    image_cv = cv2.imread(picPath)
    face = cv2.CascadeClassifier(
        "./Common/haarcascades/haarcascade_frontalface_default.xml")  # 创建人脸检测器放在同目录
    # 将image转为灰度图像，存放中gray中
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(
        gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)  # 检测图像中的人脸

    # 裁剪人脸区域，取最大的部分
    (a, b, c, d) = (0, 0, 0, 0)
    for(x, y, w, h) in faces:
        if(w*h > c*d):
            (a, b, c, d) = (x, y, w, h)
    if(b >= 40):  # 保留头发
        cropped = image_cv[b-40:b+d, a:a+c]
    else:
        cropped = image_cv[0:b+d, a:a+c]

    # 用classifier识别男性脸部的时候有可能识别不到
    if len(faces):
        croppedGrey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        croppedGrey = gray[75:175, 75:175]
        count_male += 1

    croppedGreyResize = cv2.resize(croppedGrey, (100, 100))
    cv2.imwrite(newPath+str(i), croppedGreyResize)
    time.sleep(0.01)

file_male.close()
print('male finished!', '未识别：', count_male)

# 再导入女性图片
file_female = open(r'Dataset/Label/female_names.txt', 'r')
femaleNameList = list(file_female)  # 创建女性姓名列表
for i in range(len(femaleNameList)):
    femaleNameList[i] = femaleNameList[i].strip('\n')  # 读的人名带\n，去掉之
print('Detecting female faces...')

for i in tqdm(femaleNameList):
    name = i[:i.rfind("_")]
    picPath = dataPath + 'Image' + \
        '/'+str(name)+'/'+str(i)

    # 以下进行图像的读入与裁剪
    image_cv = cv2.imread(picPath)
    face = cv2.CascadeClassifier(
        "./Common/haarcascades/haarcascade_frontalface_default.xml")  # 创建人脸检测器放在同目录
    # 将image转为灰度图像，存放中gray中
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(
        gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)  # 检测图像中的人脸

    (a, b, c, d) = (0, 0, 0, 0)
    for(x, y, w, h) in faces:   # 裁剪人脸区域
        if(w*h > c*d):
            (a, b, c, d) = (x, y, w, h)
    if(b >= 40):
        cropped = image_cv[b-40:b+d, a:a+c]
    else:
        cropped = image_cv[0:b+d, a:a+c]
    # 用classifier识别女性脸部的时候有可能识别不到
    if len(faces):
        croppedGrey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        croppedGrey = gray[75:175, 75:175]
        count_female += 1

    croppedGreyResize = cv2.resize(croppedGrey, (100, 100))
    cv2.imwrite(newPath+str(i), croppedGreyResize)
    time.sleep(0.01)

file_female.close()
print('female finished!', '未识别：', count_female)
