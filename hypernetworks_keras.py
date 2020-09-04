import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Have functions.py in the same folder
from functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('tensorflow version: {}'.format(tf.__version__))
tf.keras.backend.clear_session()

#########################
# Test on MNIST dataset
#########################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# convert to float32 and normalize. 
x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32')   /255

# one-hot encode the labels 
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# add a channel dimension to the images
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)


# Define image dimensions
img_h = 28
img_w = 28
img_c = 1

#######################
# Inference model
#######################
infer_model = tf.keras.models.Sequential(name='infer_model')
infer_model.add(tf.keras.layers.Input(shape=(img_h, img_w, img_c), name='input_x' ))
infer_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu') )
infer_model.add(tf.keras.layers.MaxPool2D() )
infer_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu') )
infer_model.add(tf.keras.layers.MaxPool2D() ) 
infer_model.add(tf.keras.layers.Flatten() )

infer_model.add(tf.keras.layers.Dense(10, activation= 'softmax', name='out_layer') )

infer_model.summary()

########################
# Hypernetwork
########################
hyper_model_x = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_h, img_w, img_c)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu') ,
        tf.keras.layers.MaxPool2D() ,
        tf.keras.layers.Conv2D(8, (3,3), activation='relu') ,
        tf.keras.layers.MaxPool2D() ,
        tf.keras.layers.Flatten() ,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=784, activation=tf.nn.relu),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=16, 
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(
            filters=8,  
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu),
        tf.keras.layers.Conv2DTranspose(
            filters=2, kernel_size=3, strides=(1, 1), padding="SAME", activation='tanh'),
        tf.keras.layers.Flatten()
    ], name='hyper_model'
)


######################
# METRICS
######################

val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3) 

loss_accum = 0.0
batch_size = 1


##########################################################
# Training loop : uses tf.GradientTape from tf-2
# Gradient Tape is more intuitive than Sessions in tf-1.x
##########################################################

for step in range(1, 6001):
  idx = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
  x, y = x_train[idx], y_train[idx]

  with tf.GradientTape() as tape:
    # Predict weights for the infer model
    generated_parameters = hyper_model_x(x)
    parameterize_model(infer_model, generated_parameters)    
    
    # Inference on the infer model
    preds = infer_model(x)

    loss = loss_fn( y, preds)
    loss_accum += loss
    train_acc_metric.update_state( y, tf.expand_dims(preds, 0)) # update the acc metric

    if step % 1000 == 0: 
      loss_accum = 0.0
      var = generated_parameters.numpy()
      print('statistics of the generated parameters: '+'Mean, {:2.3f}, var {:2.3f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))
      for val_step in range(500): # 
        idx = np.random.randint(low=0, high=x_test.shape[0], size=batch_size)
        x, y = x_test[idx], y_test[idx]
        generated_parameters = hyper_model_x(x)
        parameterize_model(infer_model, generated_parameters)    
        preds = infer_model(x)
        val_acc_metric.update_state( y, tf.expand_dims(preds, 0)) # update the acc metric
      print('\n Step: {}, validation set accuracy: {:2.2f}     loss: {:2.2f}'.format(step, float(val_acc_metric.result()), loss_accum))
      val_acc_metric.reset_states()
         
        
    # Train only hyper model
    grads = tape.gradient(loss, hyper_model_x.trainable_weights)
    optimizer.apply_gradients(zip(grads, hyper_model_x.trainable_weights))