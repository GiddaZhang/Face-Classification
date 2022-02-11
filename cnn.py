from Common.result_drawer import ResultDrawer
from Common.data_loader import TensorDataLoader
from Common.net import Model
from Common.my_callback import MyCallback


if __name__ == '__main__':

    model_num = 1  # 创建模型数量
    IMAGE_SIZE = 46  # 原图压缩后的图片尺寸

    # 实例化数据读取
    data_loader = TensorDataLoader(
        size=IMAGE_SIZE, face_centered=True, train_expanded=False)
    (train_images, train_labels), (validation_images,
                                   validation_labels), (test_images, test_labels) = data_loader.load_data()

    # 实例化网络中用到的召回
    callBack = MyCallback(patience=[3, 8])

    best_acc = 0

    for i in range(model_num):

        print('Model', i+1, 'training')
        model = Model(image_size=IMAGE_SIZE)    # 实例化网络模型
        model.summary()  # 打印网络结构

        # 训练/验证模型
        model.train(epoch_num=20, train_images=train_images, train_labels=train_labels,
                    val_images=validation_images, val_labels=validation_labels, callBack=callBack, lr=1e-3)
        # 测试模型
        accuracy = model.test(test_images, test_labels)

        if accuracy > best_acc:
            best_acc = accuracy

    # Check prediction
    '''resultPlot = ResultDrawer(row=10, col=10)
    resultPlot.plot(model, test_images, test_labels)'''

    best_acc = round(best_acc * 100, 2)
    print(f'Best accuarcy: {best_acc}%')

    '''model.test(test_images, test_labels, gender_saparated=True)'''

    # 显示界面
    model.show_tkinter()
    # model.show_gradio()
