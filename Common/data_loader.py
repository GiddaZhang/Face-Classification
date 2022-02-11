from os import name
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2


class DataLoader:

    def __init__(self):

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

    def loadPicArray(self, picFilePath):

        picData = cv2.imread(picFilePath)
        picData = cv2.resize(picData, (self.size, self.size))  # 改变图片大小
        image = Image.fromarray(cv2.cvtColor(picData, cv2.COLOR_BGR2RGB))
        image_array = np.array(image) / 255.0  # 归一化
        image_array = image_array.astype(np.float32)
        return image_array

    def loadFile(self, file_path, gender):

        with open(file_path) as file:

            for pic_name in file:
                pic_name = pic_name.rstrip()

                if self.face_centered == True:  # 此时读入人脸居中的图像
                    pic_path = self.dataPath + 'Image-haired_colored/' + pic_name
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

        self.loadFile(file_path='Dataset/Label/male_names.txt', gender='male')  # 先导入男性图片
        self.loadFile(file_path='Dataset/Label/female_names.txt', gender='female')  # 再导入女性图片

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

        print(f'Load done. Total: {self.data_count}')


class TensorDataLoader(DataLoader):

    def __init__(self, size, face_centered, train_expanded):

        # 获取训练、验证和测试数据
        super().__init__()
        self.size = size  # resize图片尺寸
        self.face_centered = face_centered  # 确认是否使用人脸居中的图像
        self.train_expanded = train_expanded  # 确认是否扩充训练集
        self.loadFaceData()

        # [train_num, width, width, 3]
        self.train_images = np.array(self.train_images)
        # [validation_num, width, width, 3]
        self.validation_images = np.array(self.validation_images)
        # [test_num, width, width, 3]
        self.test_images = np.array(self.test_images)

        self.train_labels = np.array(
            self.train_labels, dtype=np.int32)  # [train_num]
        self.validation_labels = np.array(
            self.validation_labels, dtype=np.int32)  # [validation_num]
        self.test_labels = np.array(
            self.test_labels, dtype=np.int32)  # [test_num]

        # 获得新增左右翻转的训练集
        # 左右翻转图片列元素倒序排列
        self.train_images_flip = self.train_images[:, :, ::-1, :]

        # 拼接起来
        self.train_images_plus = np.concatenate(
            [self.train_images, self.train_images_flip], axis=0)  # 四维矩阵不能直接用hstack
        self.train_labels_plus = np.hstack(
            (self.train_labels, self.train_labels))

        # 重新打乱
        state = np.random.get_state()
        np.random.shuffle(self.train_images_plus)
        np.random.set_state(state)
        np.random.shuffle(self.train_labels_plus)

    def load_data(self):

        if self.train_expanded == False:
            return (self.train_images, self.train_labels), (self.validation_images, self.validation_labels), (self.test_images, self.test_labels)
        else:
            print(
                f'train images expanded with {self.train_images_flip.shape[0]} more')
            return (self.train_images_plus, self.train_labels_plus), (self.validation_images, self.validation_labels), (self.test_images, self.test_labels)
