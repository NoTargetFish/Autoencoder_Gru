import numpy as np
import pandas as pd
import tensorflow as tf
import time 
    
def readucr(filename):#读取数据
    data = np.loadtxt(filename, dtype=str, delimiter = ',')
    Y = data[:,1:576]
    X = data[:,0:575]
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

class Autoencoder_GRU:#GRU神经网络

	def __init__(self, output_directory, input_shape, output_shape, verbose= True):#模型初始化
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, output_shape)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, output_shape):#创建模型
		input_layer = tf.keras.Input(input_shape)
		
		layer_1 = tf.keras.layers.GRU(20,return_sequences=True)(input_layer)
		layer_1 = tf.keras.layers.BatchNormalization()(layer_1)
		layer_1 = tf.keras.layers.Activation(activation='tanh')(layer_1)

		output_layer = tf.keras.layers.Dense(output_shape, activation='linear')(layer_1)

		model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='mse', optimizer = 'adam',metrics=['mae'])

		reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best1_model.hdf5'

		model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val):#训练模型 
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 16
		nb_epochs = 2000

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = tf.keras.models.load_model(self.output_directory+'best1_model.hdf5')

		y_pred = model.predict(x_val)

		tf.keras.backend.clear_session()

X_train,Y_train = readucr('Car_TRAIN.txt')
X_test,Y_test = readucr('Car_TEST.txt')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],1))

input_shape = X_train.shape[1:]

predict = Autoencoder_GRU('test',input_shape,1,1)

predict.fit(X_train,Y_train,X_test,Y_test)

#Test git 
