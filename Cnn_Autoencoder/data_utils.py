# Data loading and reprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def readucr(filename, sensors):
    # sensors 为传感器数量
    data = np.loadtxt(filename, dtype=str, delimiter=',')
    data = data[:, 0: -1]  # 去掉最后一个label
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0: truncate], dtype=np.float32)
    y = np.array(data[:, 1: (truncate + 1)], dtype=np.float32)
    timesteps = x.shape[1]
    x = x.reshape(-1, sensors, timesteps).swapaxes(2, 1)
    y = y.reshape(-1, sensors, timesteps).swapaxes(2, 1)
    print(f'{filename}: {x.shape}')
    return x, y

def readyahoo(filename):
    with open(filename, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        values = [row[1] for row in reader]
    data = np.array(values[1:], dtype=np.float32).reshape(1, len(values[1:]))
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0: truncate], dtype=np.float32)
    y = np.array(data[:, 1: (truncate + 1)], dtype=np.float32)
    x = x.reshape((x.shape[0], x.shape[1],1))
    y = y.reshape((y.shape[0], y.shape[1],1))
    print(f'{filename}: {x.shape}')
    return x, y

def Data_MinMax_Scaler(x_train): # Normoaliazation
    sample_num = x_train.shape[0]
    for i in range(0, sample_num):
        scaler = MinMaxScaler(feature_range=(-1, 1)) # 归一化到 [-1, 1] 区间内
        x_train[i] = scaler.fit_transform(x_train[i]) # fit 获得最大值和最小值，transform 执行归一化
    return x_train

def is_abnormal(modelpath, X_test, X_train, Y_train):
    mini_batch_size = 6
    flag = 0 #标识是否有异常数据
    sigma = 1.0
    model = load_model(modelpath)

    abnormal_dict = {}
    index = 0
    X_test = np.repeat(X_test, 6, axis=0)

    [threshold, _] = model.evaluate(X_train, Y_train, batch_size=mini_batch_size, verbose=0)
    predict_value = model.predict(X_test)

    threshold = threshold * sigma
    print('threshold is: {}'.format(threshold))
    [loss, _] = model.evaluate(X_test[index * 6:(index + 1) * 6], X_test[index * 6:(index + 1) * 6],
            batch_size=mini_batch_size, verbose=0)
    print('loss is: {}'.format(loss))

    # 定位异常数据时间窗
    if loss > threshold:
        print('This is an abnormal data')
        while index < X_test.shape[2]:
            j = 0
            value = 0
            while j < X_test.shape[3]:
                value = value + (X_test[0][0][index][j] - predict_value[0][0][index][j]) ** 2
                j = j + 1
            value = value / X_test.shape[3]
            print('{} of loss is: {}'.format(index,loss))
            if value > threshold:
                abnormal_dict[index] = value
            index = index + 1
        print('the index of abnormal data is: {}'.format(abnormal_dict))
        flag = 1
    else:
        print('This is a normal data')


    return abnormal_dict if flag == 1 else None