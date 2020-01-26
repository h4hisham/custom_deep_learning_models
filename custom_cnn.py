import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU, Dense, Flatten

class MyCNN(tf.keras.Model):    
    def __init__(self):
      super().__init__(name='cnn')
      #layers
      self.conv_1 = Conv2D(9, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
      self.batch_n_1 = BatchNormalization(axis=-1)
      self.leaky_1 = LeakyReLU(0.2)
      self.conv_2 = Conv2D(9, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
      self.batch_n_2 = BatchNormalization(axis=-1)
      self.leaky_2 = LeakyReLU(0.2)

      self.max_p_1 = MaxPooling2D(2, strides=2)
      self.conv_3 = Conv2D(3, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
      self.batch_n_3 = BatchNormalization(axis=-1)
      self.leaky_3 = LeakyReLU(0.2)
      self.conv_4 = Conv2D(3, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
      self.batch_n_4 = BatchNormalization(axis=-1)
      self.leaky_4 = LeakyReLU(0.2)

      self.max_p_2 = MaxPooling2D(2, strides=2)
      self.conv_5 = Conv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')

      self.flatten_1 = Flatten()
      self.dense_1 = Dense(1)
      self.output_layer = Activation('sigmoid')

    def call(self, input_tensor):
        x=self.conv_1(input_tensor)
        x=self.batch_n_1(x)
        x=self.leaky_1(x)
        x=self.conv_2(x)
        x=self.batch_n_2(x)
        x=self.leaky_2(x)

        x=self.max_p_1(x)
        x=self.conv_3(x)
        x=self.batch_n_3(x)
        x=self.leaky_3(x)
        x=self.conv_4(x)
        x=self.batch_n_4(x)
        x=self.leaky_4(x)

        x=self.max_p_2(x)
        x=self.conv_5(x)
        
        x=self.flatten_1(x)
        x=self.dense_1(x)
        return self.output_layer(x)


if __name__ == "__main__":
  import numpy as np

  cnn = MyCNN()
  output_value = cnn(np.random.uniform(0,1,size=(1,128,128,1)))
  print(output_value.numpy())