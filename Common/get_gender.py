import tkinter as tk
import tkinter.ttk as ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

photo = None
img = None


class GetGender():

    def __init__(self, model, size):
        self.model = model
        self.size = size

        # 界面
        self.root = tk.Tk()
        self.root.title('性别识别器')
        self.root.geometry('480x720')  # 这里的乘号不是 * ，而是小写英文字母 x

        # 指引文本
        self.Label_guide = tk.Label(self.root, text="输入图片路径", font=("宋体", 14))
        self.Label_guide.pack()

        # 结果文本
        self.Label_result_text = [
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar()]

        self.Label_result = [
            tk.Label(
                self.root, textvariable=self.Label_result_text[0],
                font=("宋体", 14)),
            tk.Label(
                self.root, textvariable=self.Label_result_text[1],
                font=("宋体", 14)),
            tk.Label(
                self.root, textvariable=self.Label_result_text[2],
                font=("宋体", 16, "bold"))]

        # 结果进度条
        self.progressbar = [
            ttk.Progressbar(
                self.root, orient=tk.HORIZONTAL, length=200, mode='determinate'),
            ttk.Progressbar(
                self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')]

        # 输入框
        self.entry = tk.Entry(self.root, width=30)
        self.entry.pack()

        # 显示图片
        self.label_img = tk.Label(self.root)

        def get_result():
            picPath = self.entry.get()		# 调用get()方法，将Entry中的内容获取出来

            self.showPic(picPath)  # 显示用户输入的照片

            self.showPredict(picPath)  # 显示预测结果

        self.button = tk.Button(self.root, text='识别',
                                command=get_result).pack()

        self.root.mainloop()

    def showPic(self, picFilePath):
        global photo
        global img
        img = Image.open(picFilePath)
        img = img.resize((400, 400))  # 规定图片大小
        photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=photo)
        self.label_img.pack()

    def showPredict(self, picFilePath):
        test_image = self.loadPicArray(picFilePath)
        result = []

        predict_list = self.model.predict(test_image)
        male_prob = round(predict_list[0][0]*100, 2)
        female_prob = round(predict_list[0][1]*100, 2)

        result.append(str('男性概率：') + str(male_prob) + str('%'))
        self.Label_result_text[0].set(result[0])
        self.Label_result[0].pack()

        self.progressbar[0].pack(pady=20)
        # 进度值最大值
        self.progressbar[0]['maximum'] = 100
        # 进度值初始值
        self.progressbar[0]['value'] = int(male_prob)

        result.append(str('女性概率：') + str(female_prob) + str('%'))
        self.Label_result_text[1].set(result[1])
        self.Label_result[1].pack()

        self.progressbar[1].pack(pady=20)
        # 进度值最大值
        self.progressbar[1]['maximum'] = 100
        # 进度值初始值
        self.progressbar[1]['value'] = int(female_prob)

        if male_prob >= female_prob:
            self.Label_result_text[2].set('应该是男的！')
            self.Label_result[2].pack()
        else:
            self.Label_result_text[2].set('应该是女的！')
            self.Label_result[2].pack()

    def loadPicArray(self, picFilePath):

        picData = cv2.imread(picFilePath)
        face = cv2.CascadeClassifier(
            "./Common/haarcascades/haarcascade_frontalface_default.xml")  # 创建人脸检测器放在同目录
        # 将image转为灰度图像，存放中gray中
        gray = cv2.cvtColor(picData, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(
            gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)  # 检测图像中的人脸

        # 用classifier有可能识别不到
        if len(faces):
            # 裁剪人脸区域，取最大的部分
            (a, b, c, d) = (0, 0, 0, 0)
            for(x, y, w, h) in faces:
                if(w*h > c*d):
                    (a, b, c, d) = (x, y, w, h)
            if(b >= 40):
                cropped = picData[b-40:b+d, a:a+c]  # 保留头发
            else:
                cropped = picData[0:b+d, a:a+c]

            picData = cropped

        picData = cv2.resize(picData, (self.size, self.size))  # 改变图片大小
        picData = Image.fromarray(cv2.cvtColor(picData, cv2.COLOR_BGR2RGB))
        picData = np.array(picData) / 255.0  # 归一化
        picData = np.resize(picData, (1, self.size, self.size, 3))
        return picData
