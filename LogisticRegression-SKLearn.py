from Common.data_loader import DataLoader
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    # 加载数据
    dataLoader = DataLoader(face_centered=True)
    dataLoader.loadFaceData()

    # 获取训练、验证和测试数据
    train_images, train_labels = dataLoader.getTrainData()
    validation_images, validation_labels = dataLoader.getValidationData()
    test_images, test_labels = dataLoader.getTestData()

    c = 1000  # LogisticRegression的正则化参数
    best_lr = None
    accuracy_max = 0

    print('Start training...')
    while c > 0.01:  # 通过循环尝试不同的超参数，自动找到较优的超参数

        lr = LogisticRegression(
            penalty='l1', C=c, multi_class='ovr', solver='liblinear')  # 初始化逻辑回归类对象
        lr.fit(train_images, train_labels)  # 使用训练集训练

        accuracy = 100 * lr.score(validation_images, validation_labels)
        accuracy_round = round(accuracy, 2)
        print(f'Validation accuracy: {accuracy_round}%')

        if accuracy > accuracy_max:  # 保存较优的模型
            accuracy_max = accuracy
            best_lr = lr

        c = c / 2

    print('Train done')
    accuracy_test = round(100 * best_lr.score(test_images, test_labels), 2)
    print(f'Test accuracy: {accuracy_test}%')
