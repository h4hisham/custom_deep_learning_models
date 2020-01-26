import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, UpSampling2D, Activation, LeakyReLU

class MyUnet(tf.keras.Model):
    def __init__(self, num_f=32, num_classes=1):
        super().__init__(name='Unet')
        #Layers
        self.conv_1 = Conv2D(num_f, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_1 = BatchNormalization(axis=-1)
        self.leaky_1 = LeakyReLU(0.2)
        self.drop_1 = Dropout(0.1)
        self.conv_2 = Conv2D(num_f, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_2 = BatchNormalization(axis=-1)
        self.leaky_2 = LeakyReLU(0.2)

        self.max_p_1 = MaxPooling2D(2, strides=2)
        self.conv_3 = Conv2D(num_f*2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_3 = BatchNormalization(axis=-1)
        self.leaky_3 = LeakyReLU(0.2)
        self.drop_2 = Dropout(0.1)
        self.conv_4 = Conv2D(num_f*2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_4 = BatchNormalization(axis=-1)
        self.leaky_4 = LeakyReLU(0.2)

        self.max_p_2 = MaxPooling2D(2, strides=2)
        self.conv_5 = Conv2D(num_f*4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_5 = BatchNormalization(axis=-1)
        self.leaky_5 = LeakyReLU(0.2)
        self.drop_3 = Dropout(0.1)
        self.conv_6 = Conv2D(num_f*4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_6 = BatchNormalization(axis=-1)
        self.leaky_6 = LeakyReLU(0.2)

        self.max_p_3 = MaxPooling2D(2, strides=2)
        self.conv_7 = Conv2D(num_f*8, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_7 = BatchNormalization(axis=-1)
        self.leaky_7 = LeakyReLU(0.2)
        self.drop_4 = Dropout(0.1)
        self.conv_8 = Conv2D(num_f*8, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_8 = BatchNormalization(axis=-1)
        self.leaky_8 = LeakyReLU(0.2)

        self.max_p_4 = MaxPooling2D(2, strides=2)
        self.conv_9 = Conv2D(num_f*16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_9 = BatchNormalization(axis=-1)
        self.leaky_9 = LeakyReLU(0.2)
        self.drop_5 = Dropout(0.1)
        self.conv_10 = Conv2D(num_f*16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_10 = BatchNormalization(axis=-1)
        self.leaky_10 = LeakyReLU(0.2)
        self.upsample_1 = UpSampling2D()

        self.conv_11 = Conv2D(num_f*8, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_11 = BatchNormalization(axis=-1)
        self.leaky_11 = LeakyReLU(0.2)
        self.drop_6 = Dropout(0.1)
        self.conv_12 = Conv2D(num_f*8, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_12 = BatchNormalization(axis=-1)
        self.leaky_12 = LeakyReLU(0.2)
        self.upsample_2 = UpSampling2D()

        self.conv_13 = Conv2D(num_f*4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_13 = BatchNormalization(axis=-1)
        self.leaky_13 = LeakyReLU(0.2)
        self.drop_7 = Dropout(0.1)
        self.conv_14 = Conv2D(num_f*4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_14 = BatchNormalization(axis=-1)
        self.leaky_14 = LeakyReLU(0.2)
        self.upsample_3 = UpSampling2D()

        self.conv_15 = Conv2D(num_f*2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_15 = BatchNormalization(axis=-1)
        self.leaky_15 = LeakyReLU(0.2)
        self.drop_8 = Dropout(0.1)
        self.conv_16 = Conv2D(num_f*2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_16 = BatchNormalization(axis=-1)
        self.leaky_16 = LeakyReLU(0.2)
        self.upsample_4 = UpSampling2D()

        self.conv_17 = Conv2D(num_f, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_17 = BatchNormalization(axis=-1)
        self.leaky_17 = LeakyReLU(0.2)
        self.drop_9 = Dropout(0.1)
        self.conv_18 = Conv2D(num_f, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.batch_n_18 = BatchNormalization(axis=-1)
        self.leaky_18 = LeakyReLU(0.2)

        self.conv_19 = Conv2D(num_classes, 1)
        self.output_layer = Activation('sigmoid')


    def call(self, input_tensor):
        # Definition of the forward pass
        x=self.conv_1(input_tensor)
        x=self.batch_n_1(x)
        x=self.leaky_1(x)
        x=self.drop_1(x)
        x=self.conv_2(x)
        x=self.batch_n_2(x)
        x=self.leaky_2(x)
        y1=x

        x=self.max_p_1(x)
        x=self.conv_3(x)
        x=self.batch_n_3(x)
        x=self.leaky_3(x)
        x=self.drop_2(x)
        x=self.conv_4(x)
        x=self.batch_n_4(x)
        x=self.leaky_4(x)
        y2=x

        x=self.max_p_2(x)
        x=self.conv_5(x)
        x=self.batch_n_5(x)
        x=self.leaky_5(x)
        x=self.drop_3(x)
        x=self.conv_6(x)
        x=self.batch_n_6(x)
        x=self.leaky_6(x)
        y3=x

        x=self.max_p_3(x)
        x=self.conv_7(x)
        x=self.batch_n_7(x)
        x=self.leaky_7(x)
        x=self.drop_4(x)
        x=self.conv_8(x)
        x=self.batch_n_8(x)
        x=self.leaky_8(x)
        y4=x

        x=self.max_p_4(x)
        x=self.conv_9(x)
        x=self.batch_n_9(x)
        x=self.leaky_9(x)
        x=self.drop_5(x)
        x=self.conv_10(x)
        x=self.batch_n_10(x)
        x=self.leaky_10(x)
        x=self.upsample_1(x)

        x=concatenate([x, y4], axis=-1)
        x=self.conv_11(x)
        x=self.batch_n_11(x)
        x=self.leaky_11(x)
        x=self.drop_6(x)
        x=self.conv_12(x)
        x=self.batch_n_12(x)
        x=self.leaky_12(x)
        x=self.upsample_2(x)

        x=concatenate([x, y3], axis=-1)
        x=self.conv_13(x)
        x=self.batch_n_13(x)
        x=self.leaky_13(x)
        x=self.drop_7(x)
        x=self.conv_14(x)
        x=self.batch_n_14(x)
        x=self.leaky_14(x)
        x=self.upsample_3(x)

        x=concatenate([x, y2], axis=-1)
        x=self.conv_15(x)
        x=self.batch_n_15(x)
        x=self.leaky_15(x)
        x=self.drop_8(x)
        x=self.conv_16(x)
        x=self.batch_n_16(x)
        x=self.leaky_16(x)
        x=self.upsample_4(x)

        x=concatenate([x, y1], axis=-1)
        x=self.conv_17(x)
        x=self.batch_n_17(x)
        x=self.leaky_17(x)
        x=self.drop_9(x)
        x=self.conv_18(x)
        x=self.batch_n_18(x)
        x=self.leaky_18(x)

        x=self.conv_19(x)
        return self.output_layer(x)


if __name__ == "__main__":
  import numpy as np
  import matplotlib.pyplot as plt

  unet = MyUnet(num_f=32, num_classes=1)
  output_image = unet(np.random.uniform(0,1,size=(1,128,128,1)))
  plt.imshow(tf.reshape(output_image, shape=(128,128)), cmap = "gray")