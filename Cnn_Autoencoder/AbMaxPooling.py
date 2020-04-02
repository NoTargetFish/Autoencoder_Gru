import keras
import tensorflow as tf
from keras.engine.topology import Layer


class AbMaxPooling1D(Layer):

    def __init__(self, pool_size=2, strides=None, padding='valid'):
        # self.output_dim = output_dim
        super(AbMaxPooling1D, self).__init__()
        self.maxPool1D = keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)
        self.outshape = ()

    def build(self, input_shape):
        #
        super(AbMaxPooling1D, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        output1 = self.maxPool1D(inputs)
        output2 = self.maxPool1D(-inputs)
        mask = output1 >= output2
        output = tf.where(mask, output1, -output2)
        self.outshape = output.shape
        return output

    def compute_output_shape(self, input_shape):
        return self.outshape


class AbMaxPooling2D(Layer):

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        # self.output_dim = output_dim
        super(AbMaxPooling2D, self).__init__()
        self.maxPool = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        self.outputshape = ()

    def build(self, input_shape):
        super(AbMaxPooling2D, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        output1 = self.maxPool(inputs)
        output2 = self.maxPool(-inputs)
        mask = output1 >= output2
        output = tf.where(mask, output1, -output2)
        self.outputshape = output.shape
        return output

    def compute_output_shape(self, input_shape):
        return self.outputshape


