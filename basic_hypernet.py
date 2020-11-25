import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.layers import Layer
tf.enable_eager_execution()

# The total number of parameters to predict from hyper model must be predetermined
PARAMS = 1*10 + 10 + 10*5 + 5     # kernel weights and bias terms for the dense layers in inference model

def inference_model():
    model = tf.keras.models.Sequential(name='infer_model')
    model.add(tf.keras.layers.Input(shape=(1), name='input_timestep'))    # Input is the timestep
    model.add(tf.keras.layers.Dense(10, activation='relu', name='hidden_layer_1'))
    model.add(tf.keras.layers.Dense(5, activation='relu', name='hidden_layer_2'))
    model.add(tf.keras.layers.Dense(1, activation='tanh', name='output_layer'))        # Output is the amplitude for given timestep
    # model.summary()
    return model

def hyper_model():
    # This model creates the weights of inference model
    model = tf.keras.models.Sequential(name='hyper_model')
    model.add(tf.keras.layers.Input(shape=(1), name='input_frequency'))   # Input is the frequency
    model.add(tf.keras.layers.Dense(150, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dense(100, activation='relu', name='dense_2'))
    model.add(tf.keras.layers.Dense(PARAMS, activation='tanh', name='output'))  # Output the params for inference model. Tanh is used to assign values from [-1, 1]
    return model


def parameterize_inference_model(model, weights):
    # function to parametrizes all the trainable variables of model using the stream of weight values in weights
    # This assumes weights are passed a single batch.
    weights = tf.reshape( weights, [-1] ) # reshape the parameters to a vector
    
    last_used = 0
    for i in range(len(model.layers)):
        # check to make sure only non output fully connected layers are assigned weights.
        if 'hidden' in model.layers[i].name: 
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            model.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if model.layers[i].use_bias:
              weights_shape = model.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              model.layers[i].bias = new_weights
              last_used += no_of_weights



for layer in inference_model().layers:
    print(layer.kernel.shape, layer.name, int(tf.reduce_prod(layer.kernel.shape)))


# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

  
loss_accum = 0.0
batch_size = 1
# Ranges from 20 to 40 hz
frequencies = list(range(20,40))
# Time period is restricted to 1s with a sampling rate of 100
time_periods = np.random.sample((100,))
dataset = []
for f in frequencies:
    for t in time_periods:
        dataset.append((f, t, np.sin(f*t)))
size = len(dataset)
print(size)
for step in range(1, 1001):
  # Put dataset here : We fit a curve for A*sin(ft), A=1; f=frequency; t= time period
    idx = np.random.randint(low=0, high=size, size=batch_size)[0]
    # print(step, idx)
    f = np.array([dataset[idx][0]]).reshape((1, -1))
    t = np.array([dataset[idx][1]]).reshape((1, -1))
    y = np.array([dataset[idx][2]]).reshape((1, -1))
    with tf.GradientTape() as tape:

        # Predict weights for the outer model.
        hyper = hyper_model()
        generated_parameters = hyper(f)
        infer_model = inference_model()
        parameterize_inference_model(infer_model, generated_parameters)

        # Predict from inference model
        preds = infer_model(t)
        loss = loss_fn(y, preds)
        loss_accum += loss

        # Train only inner model.

        print("Loss : ", loss)
        if step%100 == 0:
            print("Loss : ", loss)
            var = generated_parameters.numpy()
            print('statistics of the generated parameters: '+'Mean, {:2.3f}, var {:2.3f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))
    grads = tape.gradient(loss, hyper.trainable_weights)
    optimizer.apply_gradients(zip(grads, hyper.trainable_weights))