import numpy as np
import pandas as pd
import keras 
import time 
    
def readucr(filename):#读取数据
    data = np.loadtxt(filename, dtype=str, delimiter = ',')
    Y = data[:,1:576]
    X = data[:,0:575]
    return X, Y

class Classifier_GRU:#GRU神经网络

	def __init__(self, output_directory, input_shape, output_shape, verbose=False):#模型初始化
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, output_shape)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, output_shape):#创建模型
		input_shape = input_shape[0].shape
		input_layer = keras.layers.Input(input_shape)
		
		layer_1 = keras.layers.GRU(20,return_sequences=True)(input_layer)
		layer_1 = keras.layers.normalization.BatchNormalization()(layer_1)
		layer_1 = keras.layers.Activation(activation='tanh')(layer_1)

		output_layer = keras.layers.Dense(output_shape, activation='linear')(layer_1)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='mse', optimizer = keras.optimizers.Adam(),
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
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

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		keras.backend.clear_session()

X_train,Y_train = readucr('Car_TRAIN.txt')

X_test,Y_test = readucr('Car_TEST.txt')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],1))

Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],1))

input_shape = X_train.shape[1:]

classifier = Classifier_GRU('test',input_shape,1,True)

classifier.fit(X_train,Y_train,X_test,Y_test)

#Test git 
