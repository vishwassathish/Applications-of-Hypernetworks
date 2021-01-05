import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data
tf.enable_eager_execution()

# Directory to print results to
results_dir = "./results/"

# Hyperparameters
N_img = 8                       # Number of cosine images
batch_size = 32                 # Batch Size

k_sz = 3                        # Filter Size
epochs = 1000000                # Number of epochs to train for

embed_model_hidden_1 = 96       # # of parameters for the first embed hidden
embed_model_hidden_2 = 64       # # of parameters for the second embed hidden
embed_model_output = 16         # Embedding dimension

cos_model_hidden_1 = 64         # # of parameters for the first cos hidden
cos_model_hidden_2 = 64         # # of parameters for the second cos hidden
cos_model_output = 1            # Predicition dimension

lr = 5e-4                       # Learning Rate
hist_size = 1e3                 # Size of histogram

# Generate Dataset
data, f = generate_data(num_examples=N_img)
im_H, im_W = data.shape[1:3]


# Define Models
# Generate embeddings from image
def embedding_model():
    model = tf.keras.models.Sequential(name='embed_model')
    model.add(tf.keras.layers.Input(shape=(im_H, im_W, 1), name='input_image'))                 # Input is the timestep
    model.add(tf.keras.layers.Conv2D(1, k_sz, activation='relu', name='hidden_layer_1'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(embed_model_hidden_1, activation='elu', name='dense_1'))
    model.add(tf.keras.layers.Dense(embed_model_hidden_2, activation='elu', name='dense_2'))
    model.add(tf.keras.layers.Dense(embed_model_output, name='output_layer'))                   # Output is the amplitude for given timestep
    return model


# Generate cos(2pift) from embedding and frequency
def cos_model():
    # This model creates the weights of inference model
    model = tf.keras.models.Sequential(name='cos_model')
    model.add(tf.keras.layers.Input(shape=(embed_model_output + 1), name='input_embedding_timestep'))   # Input is the frequency
    model.add(tf.keras.layers.Dense(cos_model_hidden_1, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dense(cos_model_output, activation='tanh', name='output'))                # Output the params for inference model. Tanh is used to assign values from [-1, 1]
    return model


# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Initialize logging information
loss_accum = 0.0
hist_loss = []
loss = 0
verbosity = 500

# Initialize models
embedding = embedding_model()
cos = cos_model()

# Train
for epoch in range(epochs):
    # Batch Everything
    idx = np.random.randint(0, N_img, batch_size)
    batch_I = data[idx, :, :]
    batch_t = np.float32((np.random.randint(0, 100, batch_size) / 100).reshape([-1, 1]))
    batch_y = np.sin(2*np.pi*(idx+1).reshape([-1, 1])*batch_t)

    with tf.GradientTape() as tape:
        # Predict embedding and concatenate t
        z = embedding(batch_I)
        cos_input = tf.keras.layers.concatenate([batch_t, z], axis=1)

        # Predict from inference model
        preds = cos(cos_input)

        # Loss
        loss = loss_fn(batch_y, preds)
        loss_accum += loss

    # Update weights
    variables = embedding.trainable_variables + cos.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    # To get number of parameters:
    #   print(np.sum([np.prod(v.shape) for v in variables]))

    # Log
    if epoch % verbosity == 0:
        # Timesteps to show
        comb = np.float32(np.linspace(0, 1, 200).reshape([-1, 1]))

        # Groundtruth Frequency
        idx = np.random.randint(0, N_img, 1)
        ground_f = np.float32(idx) + 1

        # Batch images
        batch_I = data[idx, :, :]
        batch_I = np.tile(batch_I, [comb.shape[0], 1, 1, 1])

        # Predict embedding and concatenate t
        z = embedding(batch_I)
        cos_input = tf.keras.layers.concatenate([comb, z], axis=1)

        # Predict from inference model
        preds = cos(cos_input)

        # Generate ground truth
        SIN = np.sin(2*np.pi*ground_f*comb)

        # Plot prediction and ground truth
        plt.plot(comb, preds)
        plt.plot(comb, SIN, color='r')
        plt.savefig(results_dir + 'sin_hat.png')
        plt.clf()

        # Report Loss
        print("Loss {:2}: {:2.3f}".format(epoch, loss.numpy()))
        hist_loss += [loss.numpy()]

        # Plot Loss
        plt.plot(np.array(hist_loss))
        plt.savefig(results_dir + 'loss.png')
        plt.clf()

        # Refresh hist if needed
        if len(hist_loss) > hist_size:
            hist_loss = hist_loss[-hist_size:]
