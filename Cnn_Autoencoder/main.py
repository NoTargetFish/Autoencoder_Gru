from data_utils import readucr, Data_MinMax_Scaler, is_abnormal
from  cnn_AE_1 import Cnn_AE_1
from cnn_AE_1pro import Cnn_AE_1pro
from GRU import Autoencoder_GRU, output_of_gru
from cnn_AE_2 import Cnn_AE_2

import numpy as np
import sys
import os


if __name__ == '__main__':
    # 读取数据
    #dataset_name = sys.argv[1]
    dataset_name = 'Car'
    dataset_path = os.path.join('data', dataset_name)
    x_train, y_train = readucr(dataset_path + '/' + dataset_name + '_TRAIN_Order.txt', 4)
    x_test, y_test = readucr(dataset_path + '/' + dataset_name + '_TEST_Order.txt', 4)

    # 将数据归一化
    x_train = Data_MinMax_Scaler(x_train)
    y_train = Data_MinMax_Scaler(y_train)
    x_test = Data_MinMax_Scaler(x_test)
    y_test = Data_MinMax_Scaler(y_test)

    # 划分 训练集 和 验证集
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
    #                                                       test_size=0.33,
    #                                                       shuffle=True,
    #                                                       random_state=42)

    # 训练gru模型
    gru_model = Autoencoder_GRU(dataset_name, x_train.shape[1:], x_train.shape[2])
    gru_modelpath = gru_model.model_path
    gru_model.fit(x_train, y_train, x_test, y_test, epochs=2)

    # 提取gru的输出
    gru_train = output_of_gru(x_train, gru_modelpath)
    gru_test = output_of_gru(x_test, gru_modelpath)

    # 训练cnn模型
    input_shape = (1, gru_train.shape[2], gru_train.shape[3])
    cnn_Auto = Cnn_AE_1pro(dataset_name, input_shape, True)
    cnn_Auto.fit_model(gru_train, gru_train, gru_test, gru_test, epochs=2)
    cnn_modelpath = cnn_Auto.model_path

    # 输入测试数据并将其归一化
    # test_data, _ = readucr(dataset_path + '/' + 'test_data.txt')
    # test_data = Data_MinMax_Scaler(test_data)
    # g_test = output_of_gru(test_data, gru_modelpath)
    # print(g_test.shape)
    # 测试是否异常
    #gru_train = gru_train[:3600,:]
    # is_abnormal(cnn_modelpath, g_test[:1], gru_test, gru_test)



