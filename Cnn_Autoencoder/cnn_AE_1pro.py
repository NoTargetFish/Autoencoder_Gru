# Import all the required Libraries
import os
import time
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Input, \
     UpSampling2D, ZeroPadding1D, ZeroPadding2D, Lambda, Conv2DTranspose, \
     Activation, Concatenate, GaussianNoise

# tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    # output1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    # output2 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
    output1 = tf.nn.max_pool1d(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2 = tf.nn.max_pool1d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    return output

def abMaxPooling2D(inputs, pool_size=[2, 2], strides=2, padding='SAME'):
    # output1 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    # output2 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
    output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    return output


def abMaxPooling_with_argmax(inputs, pool_size=2, strides=2, padding='SAME'):

    output1, argmax1 = tf.nn.max_pool_with_argmax(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2, argmax2 = tf.nn.max_pool_with_argmax(-inputs, ksize=pool_size, strides=strides, padding=padding)
    argmax1 = tf.stop_gradient(argmax1)
    argmax2 = tf.stop_gradient(argmax2)
    # output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
    # output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    argmax = tf.where(mask, argmax1, argmax2)
    return (output, argmax)

def unAbMaxPooling(inputs_argmax, ksize, strides=2, padding='SAME'):
    # 假定 ksize = strides
    inputs = inputs_argmax[0]
    argmax = inputs_argmax[1]
    input_shape = inputs.get_shape()
    if padding == 'SAME':
        rows = input_shape[1] * ksize[1]
        cols = input_shape[2] * ksize[2]
    else:
        rows = (input_shape[1]-1) * ksize[1] + ksize[1]
        cols = (input_shape[2]-1) * ksize[2] + ksize[2]
    # 计算new shape
    output_shape = (input_shape[0], rows, cols, input_shape[3])
    # 计算索引
    one_like_mask = tf.ones_like(argmax)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = argmax // (output_shape[2] * output_shape[3])
    x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    c = one_like_mask * feature_range
    # 转置索引
    update_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, update_size]))
    values = tf.reshape(inputs, [update_size])
    outputs = tf.scatter_nd(indices, values, output_shape)
    return outputs


# def reshapes(x, retype):
#     if retype == 'reshapedim':
#         x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
#     if retype == 'squeeze':
#         x = tf.squeeze(x, [1])
#     return x

def reshapes(x):
    x = tf.squeeze(x, [1])
    x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
    return x

# def reshape_output_shape(input_shape):
#     if len(input_shape) == 4:
#         return (input_shape[0], input_shape[2], input_shape[3])
#     if len(input_shape) == 3:
#         return (input_shape[0], input_shape[2], input_shape[1], 1)


class Cnn_AE_1pro:
    def __init__(self, dataset_name, input_shape, verbose=False):
        self.dataset_name = dataset_name
        self.model = self.build_model(input_shape)
        # verbose是信息展示模式
        if verbose == True:
            self.model.summary()
        self.verbose = verbose

    def build_model(self, input_shape):
        # input --> (None, 1, 576, 1)
        input_layer = Input(batch_shape=(6, input_shape[0], input_shape[1], input_shape[2]))
        # Encoder
        # conv block -1 （卷积+池化）
        h1 = input_layer.shape[1]
        # inception1
        conv1_incep1 = Conv2D(filters=4, kernel_size=(h1, 1))(input_layer)
        conv1_incep1 = BatchNormalization()(conv1_incep1)
        conv1_incep1 = Activation(activation='relu')(conv1_incep1)
        # inception2
        conv1_incep2 = ZeroPadding2D((0, 1))(input_layer)
        conv1_incep2 = Conv2D(filters=6, kernel_size=(h1, 3))(conv1_incep2)
        conv1_incep2 = BatchNormalization()(conv1_incep2)
        conv1_incep2 = Activation(activation='relu')(conv1_incep2)
        # inception3
        conv1_incep3 = ZeroPadding2D((0, 2))(input_layer)
        conv1_incep3 = Conv2D(filters=6, kernel_size=(h1, 5))(conv1_incep3)
        conv1_incep3 = BatchNormalization()(conv1_incep3)
        conv1_incep3 = Activation(activation='relu')(conv1_incep3)
        # concat
        conv1 = Concatenate(axis=-1)([conv1_incep1, conv1_incep2, conv1_incep3])
        # 加高斯噪声
        # conv1 = GaussianNoise(stddev=0.1)(conv1)
        # 池化层
        conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool1')(conv1)
        # conv1_pool = Lambda(abMaxPooling2D, arguments={'pool_size': [1, 2]}, name='abMaxPool1')(conv1)
        conv1_pool = Lambda(reshapes, name='reshape1')(conv1_pool)


        # conv block -2 （卷积+池化）
        h2 = conv1_pool.shape[1]
        # inception1
        conv2_incep1 = Conv2D(filters=2, kernel_size=(h2, 1))(conv1_pool)
        conv2_incep1 = BatchNormalization()(conv2_incep1)
        conv2_incep1 = Activation(activation='relu')(conv2_incep1)
        # inception2
        conv2_incep2 = ZeroPadding2D((0, 1))(conv1_pool)
        conv2_incep2 = Conv2D(filters=3, kernel_size=(h2, 3))(conv2_incep2)
        conv2_incep2 = BatchNormalization()(conv2_incep2)
        conv2_incep2 = Activation(activation='relu')(conv2_incep2)
        # inception3
        conv2_incep3 = ZeroPadding2D((0, 2))(conv1_pool)
        conv2_incep3 = Conv2D(filters=3, kernel_size=(h2, 5))(conv2_incep3)
        conv2_incep3 = BatchNormalization()(conv2_incep3)
        conv2_incep3 = Activation(activation='relu')(conv2_incep3)
        # concat
        conv2 = Concatenate(axis=-1)([conv2_incep1, conv2_incep2, conv2_incep3])
        # 加高斯噪声
        # conv2 = GaussianNoise(stddev=0.1)(conv2)
        # 池化层
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool2')(conv2)
        # conv2_pool = Lambda(abMaxPooling2D, arguments={'pool_size': [1, 2]}, name='abMaxPool2')(conv2)
        conv2_pool = Lambda(reshapes, name='reshape2')(conv2_pool)

        # conv block -3 （卷积）
        h3 = conv2_pool.shape[1]
        # inception1
        conv3_incep1 = Conv2D(filters=2, kernel_size=(h3, 1))(conv2_pool)
        conv3_incep1 = BatchNormalization()(conv3_incep1)
        conv3_incep1 = Activation(activation='relu')(conv3_incep1)
        # inception2
        conv3_incep2 = ZeroPadding2D((0, 1))(conv2_pool)
        conv3_incep2 = Conv2D(filters=3, kernel_size=(h3, 3))(conv3_incep2)
        conv3_incep2 = BatchNormalization()(conv3_incep2)
        conv3_incep2 = Activation(activation='relu')(conv3_incep2)
        # inception3
        conv3_incep3 = ZeroPadding2D((0, 2))(conv2_pool)
        conv3_incep3 = Conv2D(filters=3, kernel_size=(h3, 5))(conv3_incep3)
        conv3_incep3 = BatchNormalization()(conv3_incep3)
        conv3_incep3 = Activation(activation='relu')(conv3_incep3)
        # concat
        conv3 = Concatenate(axis=-1)([conv3_incep1, conv3_incep2, conv3_incep3])
        # 加高斯噪声
        # conv3 = GaussianNoise(stddev=0.1)(conv3)
        # 池化
        encoder, conv3_argmax  = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool3')(conv3)
        # encoder = Lambda(abMaxPooling2D, arguments={'pool_size': [1, 2]}, name='abMaxPool3')(conv3)


        # decoder
        # conv block -1 （反卷积+反池化）
        # deconv1_unpool = UpSampling2D(size=(1, 2))(encoder)
        deconv1 = Conv2DTranspose(filters=8, kernel_size=(h3, 3), padding='same')(encoder)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool1')([deconv1, conv3_argmax])
        # deconv1 = ZeroPadding2D((0, 1))(deconv1_unpool)
        # deconv1 = Conv2D(filters=8, kernel_size=(h3, 3), activation='relu')(deconv1)

        # conv block -2 （反卷积+反池化）
        # deconv2_unpool = UpSampling2D(size=(1, 2))(deconv1)
        deconv2 = Conv2DTranspose(filters=8, kernel_size=(h2, 3), padding='same')(deconv1_unpool)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(activation='relu')(deconv2)
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool2')([deconv2, conv2_argmax])
        # deconv2 = ZeroPadding2D((0, 1))(deconv2_unpool)
        # deconv2 = Conv2DTranspose(filters=8, kernel_size=(h2, 3), padding='same', activation='relu')(deconv2)

        # conv block -3 （反卷积+反池化）
        # deconv3_unpool = UpSampling2D(size=(1, 2))(deconv2)
        deconv3 = Conv2DTranspose(filters=16, kernel_size=(h1, 3), padding='same')(deconv2_unpool)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(activation='relu')(deconv3)
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool3')([deconv3, conv1_argmax])

        # decoder = ZeroPadding2D((0, 1))(deconv3)
        if input_shape[0] == 1:
            output_layer = Conv2DTranspose(filters=input_shape[2], kernel_size=(deconv3.shape[1], 3), padding='same')(
                deconv3_unpool)
            output_layer = BatchNormalization()(output_layer)
            output_layer = Activation(activation='tanh')(output_layer)
        else:
            output_layer = Conv2DTranspose(filters=input_shape[0], kernel_size=(deconv3.shape[1], 3), padding='same')(
                deconv3_unpool)
            output_layer = BatchNormalization()(output_layer)
            output_layer = Activation(activation='tanh')(output_layer)
            output_layer = Lambda(reshapes, name='reshape3')(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(0.001),
                      metrics=['mse'],
                      experimental_run_tf_function=False)

        now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_path = 'result' + '/cnn_AE_1pro/' + self.dataset_name + now_time + '_cnn_model.hdf5'

        log_dir = "logs/cnn/" + now_time
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                           monitor='val_loss',
                                                           save_best_only=True,
                                                           mode='auto')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=20,
                                                      min_lr=0.0001)

        self.callbacks = [tensorboard, model_checkpoint, reduce_lr]

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, epochs=400):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # batch_size = 12

        # 小批量训练大小
        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        mini_batch_size = 6

        # 开始时间
        start_time = time.time()

        # 训练模型
        hist = self.model.fit(x_train, y_train,
                              epochs=epochs,
                              verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks)

        # print('history：', hist.history)
        # 训练持续时间
        duration = time.time() - start_time

        print('duration: ', duration)

        # 做测试，所以需要加载模型
        #model = load_model(file_path)

        #loss = model.evaluate(x_train, y_train, batch_size=mini_batch_size, verbose=0)
        # y_pred = model.predict(x_val)
        # y_pred = np.argmax(y_pred, axis=1)
        #print('train_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)
        keras.backend.clear_session()