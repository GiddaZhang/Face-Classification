from Common.k_neighbors_classifier import KNeighborsClassifier
from Common.data_loader import DataLoader
import time


if __name__ == '__main__':

    # 初始化：定义超参数范围，记录开始时间
    K_min = 1
    K_max = 15
    time_start = time.time()

    def printTime():
        t = (time.time()-time_start)/60
        print('time=', ('%.2f' % t), 'min')

    # 加载数据
    printTime()
    dataLoader = DataLoader()
    dataLoader.loadFaceData()
    printTime()

    # 获取训练、验证和测试数据
    train_images, train_labels = dataLoader.getTrainData()
    validation_images, validation_labels = dataLoader.getValidationData()
    test_images, test_labels = dataLoader.getTestData()

    best_knn = None
    accuracy_max = 0

    print('Start training...')
    for k in range(K_min, K_max+1, 2):  # 通过循环尝试不同的超参数，自动找到较优的超参数
        knn = KNeighborsClassifier()  # 初始化KNN类对象
        knn.train(train_images, train_labels, k)  # 使用训练集训练
        result = knn.predict(
            validation_images, validation_labels)  # 使用验证集预测
        accuracy = result * 100

        if accuracy > accuracy_max:  # 保存较优的模型
            accuracy_max = accuracy
            best_knn = knn

        accuracy_round = round(accuracy, 2)
        print(f'Validation accuracy: {accuracy_round}%')
        printTime()

    print('Start testing...')
    result = best_knn.predict(
        test_images, test_labels)  # 使用测试集预测最终结果
    accuracy = round(result * 100, 2)

    print(f'Test accuracy: {accuracy}%')
    printTime()
