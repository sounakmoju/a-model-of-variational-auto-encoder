import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.losses import binary_crossentropy

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
latent_dim=2
(input_train, target_train), (input_test, target_test) = mnist.load_data()

img_width, img_height = input_train.shape[1], input_train.shape[2]
batch_size = 128
no_epochs = 100
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1

input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
input_test = input_test.reshape(input_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)


input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

input_train = input_train / 255
input_test = input_test / 255

i=Input(shape=input_shape,name="encoder_input")
cx= Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
cx= BatchNormalization()(cx)
cx= Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx= BatchNormalization()(cx)
x= Flatten()(cx)
mu= Dense(latent_dim, name='latent_mu')(x)
sigma=Dense(latent_dim, name='latent_sigma')(x)
encoder=Model(i,(mu, sigma), name="Encoder")
flt_size=K.size(x)
conv_shape=K.int_shape(cx)

z=Lambda(sampling_eps,name="encoder_output")([mu,sigma])
encoder1=Model([mu,sigma],z,name="encoser_2")

def sampling_eps(args):
    mu,sigma=args
    eps=K.random_normal(shape=K.shape(mu),mean=0,stddev=1)
    z=mu+K.exp(sigma/2)
    return z
d_i=Input(shape=(latent_dim,),name="decoder_input")
x=Dense(flt_size,name="dense")(d_i)
x=Reshape(conv_shape,name="reshape_layer")(x)
cx=Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
cx=BatchNormalization()(cx)
cx=Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
cx=BatchNormalization()(cx)
o=Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
decoder=Model(d_i,o,name="decoder1")

O1=encoder(i)
O2=encoder1(O1)
O3=decoder(O2)
vae=Model(i,O3,name="autoencoder")
optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
def mse(true,pred):
    loss_mse=K.mean(K.square(true-pred),axis=[1,2,3])
    return 1000*loss
def K1_loss(mu,sigma):
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(sigma), axis = 1)
    return k1_loss
def overall_loss(true,pred,mu,sigma):
    r_loss=mse(true,pred)
    k_loss=k1_loss(mu,sigma)
    return r_loss+k_loss
vae.compile(optimizer, loss=overall_loss)
vae.fit(input_train, batch_size=64,epochs=2,validation_data=(input_test))




