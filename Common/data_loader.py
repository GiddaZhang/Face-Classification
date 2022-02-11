from os import name
from PIL import Image
import numpy as np


class DataLoader:

    def __init__(self, face_centered):

        self.images = []  # 图像汇总
        self.labels = []  # 男性：编码0；女性：编码1

        self.train_images = []  # 训练用图像
        self.train_labels = []  # 训练用标签

        self.validation_images = []  # 验证用图像
        self.validation_labels = []  # 验证用标签

        self.test_images = []  # 测试用图像
        self.test_labels = []  # 测试用标签

        self.data_count = 0  # 记录样本总量
        self.dataPath = 'Dataset/'
        self.face_centered = face_centered
        print('Loading Face data...')

    def loadPicArray(self, picFilePath):

        picData = Image.open(picFilePath)
        picDataGrey = picData.convert('L')  # 转成灰色
        picArray = np.array(picDataGrey).flatten() / 255.0  # 分量为图片各像素灰度的行向量
        return picArray

    def loadFile(self, file_path, gender):

        with open(file_path) as file:

            for pic_name in file:
                pic_name = pic_name.rstrip()

                if self.face_centered == True:  # 此时读入人脸居中的图像
                    pic_path = self.dataPath + 'Image_face-centered/' + pic_name
                else:  # 读入原始图像
                    person_name = pic_name[:pic_name.rfind('_')]
                    pic_path = self.dataPath + 'Image' + \
                        '/'+person_name+'/' + pic_name

                image = self.loadPicArray(pic_path)
                self.images.append(image)
                self.data_count += 1

                if gender == 'male':
                    self.labels.append(0)  # 男性：编码0
                elif gender == 'female':
                    self.labels.append(1)  # 女性：编码1

    def loadFaceData(self):

        self.loadFile('Dataset/Label/male_names.txt', 'male')  # 先导入男性图片
        self.loadFile('Dataset/Label/female_names.txt', 'female')  # 再导入女性图片

        # 打乱数据，使用相同的次序打乱images和labels，保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        # 按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.data_count * 0.4)
        validationIndex = int(self.data_count * 0.5)
        self.train_images = self.images[0: trainIndex]
        self.train_labels = self.labels[0: trainIndex]
        self.validation_images = self.images[trainIndex: validationIndex]
        self.validation_labels = self.labels[trainIndex: validationIndex]
        self.test_images = self.images[validationIndex:]
        self.test_labels = self.labels[validationIndex:]

        print(f'Load done, total: {self.data_count}')

    def getTrainData(self):
        return self.train_images, self.train_labels

    def getValidationData(self):
        return self.validation_images, self.validation_labels

    def getTestData(self):
        return self.test_images, self.test_labels
