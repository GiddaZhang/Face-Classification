from Common.data_loader import DataLoader
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    # 加载Face数据
    dataLoader = DataLoader(face_centered=True)
    dataLoader.loadFaceData()

    # 获取训练、验证和测试数据
    train_images, train_labels = dataLoader.getTrainData()
    validation_images, validation_labels = dataLoader.getValidationData()
    test_images, test_labels = dataLoader.getTestData()

    best_knn = None
    accuracy_max = 0

    print('Start training...')
    for k in range(1, 16, 2):  # 通过循环尝试不同的超参数，自动找到较优的超参数
        knn = KNeighborsClassifier(n_neighbors=k)  # 初始化KNN类对象
        knn.fit(train_images, train_labels)  # 使用训练集训练
        accuracy = round(100 * knn.score(validation_images,
                                         validation_labels), 2)  # 使用验证集预测
        print(f'Validation accuracy: {accuracy}%')
        if accuracy > accuracy_max:  # 保存较优的模型
            accuracy_max = accuracy
            best_knn = knn
    print('Train done, testing:')

    accuracy = round(100 * best_knn.score(test_images,
                     test_labels), 2)  # 使用测试集预测最终结果
    print(f'Test accuracy: {accuracy}%')
