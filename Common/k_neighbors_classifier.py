import operator
import numpy as np


class KNeighborsClassifier:

    data_x = None
    data_y = None
    k = 1

    # 训练函数
    #data_x: 训练数据集合
    #data_y: 训练数据标签集合
    #k: K近邻分类器参数k
    def train(self, data_x, data_y, k):
        self.data_x = data_x
        self.data_y = data_y
        self.k = k

    # 预测函数
    #data_x: 训练数据
    #data_y: 训练数据标签
    def predict(self, data_x, data_y):
        train_data_count = len(self.data_x)
        testDataCount = len(data_x)
        errorCount = 0
        guess_male_num = 0
        guess_female_num = 0
        guess_male_err = 0
        guess_female_err = 0
        for i in range(len(data_x)):  # 逐个计算预测数据的分类
            data_x_one = data_x[i]
            data_y_one = data_y[i]
            test_rep_mat = np.tile(data_x_one, (train_data_count, 1))
            diff_mat = test_rep_mat - self.data_x
            sq_diff_mat = diff_mat**2
            sq_dist = sq_diff_mat.sum(axis=1)
            distance = sq_dist**0.5
            dist_index = distance.argsort()
            class_count = {}
            for k_i in range(self.k):  # 统计距离训练数据中最近的K个
                label = self.data_y[dist_index[k_i]]
                class_count[label] = class_count.get(label, 0) + 1
            class_count_list = sorted(class_count.items(), key=operator.itemgetter(
                1), reverse=True)  # 用最近的K个的标签进行投票
            if class_count_list[0][0] == 0:
                guess_male_num += 1
            else:
                guess_female_num += 1
            if class_count_list[0][0] != data_y_one:  # 统计错误个数
                errorCount = errorCount + 1
                if class_count_list[0][0] == 0:
                    guess_male_err += 1
                else:
                    guess_female_err += 1

        error_rate = float(errorCount) / testDataCount  # 计算错误率
        accuracy = 1.0-error_rate
        error_rate_male = float(guess_male_err) / guess_male_num
        error_rate_female = float(guess_female_err) / guess_female_num

        return accuracy, error_rate_male, error_rate_female  # 返回错误率
