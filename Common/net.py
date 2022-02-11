import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import gradio as gr
from Common.get_gender import GetGender


class Model(object):

    def __init__(self, image_size):
        self.size = image_size
        # 定义网络结构
        self.model = tf.keras.Sequential()

        self.model.add(
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(self.size, self.size, 3)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2))

        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))

        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

    def train(self, epoch_num, train_images, train_labels, val_images, val_labels, callBack, lr):
        adam = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=adam,
                           loss=tf.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
        self.model.fit(train_images, train_labels,
                       epochs=epoch_num,
                       validation_data=(val_images, val_labels),
                       callbacks=[callBack],
                       verbose=1)

    def test(self, test_images, test_labels, gender_saparated=False):

        if gender_saparated == False:  # 男性女性合并测试
            print('testing...')
            loss, accuracy = self.model.evaluate(
                test_images, test_labels, verbose=1)
            return accuracy

        else:  # 男性女性分开测试
            test_images_male = []
            test_images_female = []
            for i in range(len(test_labels)):
                if test_labels[i] == 0:
                    test_images_male.append(test_images[i])
                else:
                    test_images_female.append(test_images[i])

            test_labels_male = np.zeros(len(test_images_male))
            test_labels_female = np.ones(len(test_images_female))
            test_images_male = np.array(test_images_male)
            test_images_female = np.array(test_images_female)

            print('Male testing...')
            self.model.evaluate(test_images_male, test_labels_male, verbose=1)
            print('Female testing...')
            self.model.evaluate(test_images_female,
                                test_labels_female, verbose=1)

    def predict(self, predict_images):
        return self.model.predict(predict_images)

    def summary(self):
        return self.model.summary()

    def show_gradio(self):

        def predict(inp):

            label = ['男', '女']

            face = cv2.CascadeClassifier(
                "./Common/haarcascades/haarcascade_frontalface_default.xml")  # 创建人脸检测器放在同目录
            gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(
                gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)  # 检测图像中的人脸

            (a, b, c, d) = (0, 0, 0, 0)
            for(x, y, w, h) in faces:   # 裁剪人脸区域
                if(w*h > c*d):
                    (a, b, c, d) = (x, y, w, h)

            if len(faces):
                inp = inp[b:b+d, a:a+c]

            inp = cv2.resize(inp, (self.size, self.size))
            inp = Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
            inp = np.array(inp) / 255.0  # 归一化
            inp = np.resize(inp, (1, self.size, self.size, 3))

            predict_list = self.model.predict(inp)

            return {label[i]: float(predict_list[0][i]) for i in range(2)}

        inputs = gr.inputs.Image()
        outputs = gr.outputs.Label(num_top_classes=2)
        gr.Interface(fn=predict, inputs=inputs,
                     outputs=outputs).launch(inbrowser=True)

    def show_tkinter(self):
        get_gender = GetGender(self.model, self.size)
