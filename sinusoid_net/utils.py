import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image


# Convert rgb image to grayscale
def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.144])


# Function to generate example images
#   num_examples: (100) Number of images to generate
#   random_f: (False) If True generate fs, else f = np.arange(num_examples)
#   min: (0.0) If random_f is True, minimum value for f
#   max: (1.0) If random_f is True, maximum value for f
#   sampling_rate: (100) Rate at which to sample for image
#   np_fnm: (None) If None, no .npy file for data, else filename
#   verbosity: (0) If nonzero, log update rate
def generate_data(num_examples=100, random_f=False, min=0.0, max=1.0,
                  sampling_rate=100, np_fnm=None, verbosity=0):
    # Saving location
    temp_dir = './temp/'
    fnm = temp_dir + 'sin.png'

    # Hyperparameters
    out_H = out_W = 32  # Size of image

    # Generate timesteps
    t = np.linspace(0, 1.0, sampling_rate)
    t_tile = np.tile(t, (num_examples, 1))

    # Generate f
    if random_f:
        f = np.random.uniform(min, max, num_examples)
    else:
        f = np.arange(num_examples) + 1

    # Generate labels
    y = np.sin(2 * np.pi * t_tile * f[:, None])

    # Generate images
    images = []
    for i in range(num_examples):
        # Temporary
        plt.plot(t, np.squeeze(y[i:i+1, :]))
        plt.axis('off')
        plt.savefig(fnm, bbox_inches='tight')
        plt.clf()

        # Grab image from plt
        img = np.asarray(Image.open(fnm))
        img = rgb2gray(img / 255.0)

        # Clean images
        h_border = 10
        w_border = 10
        img = img[h_border:-h_border, w_border:-w_border]
        img = Image.fromarray(img).resize((out_H, out_W),
                                          resample=Image.BICUBIC)
        img = np.expand_dims(np.asarray(img), axis=-1)
        images.append(img)

        # Logging
        if verbosity > 0 and i % verbosity == 0:
            print("Generating Datapoint: " + str(i))

    # If we want to save as .npy
    images = np.asarray(images)
    if np_fnm is not None:
        np.save(np_fnm, images)
        if verbosity > 0:
            print("Saved data here: " + str(np_fnm))

    # Example image
    plt.imshow(images[0], cmap="gray")
    plt.savefig(fnm)
    plt.clf()
    return images, f
