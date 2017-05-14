# keras model for behavioual cloning project
# Talaat Khalil

# importing the needed libraries
import numpy as np
from scipy import misc
import argparse
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# read driving data functions

# read the lines function
def read_data_lines(data_path):
    log_path = data_path + '/driving_log.csv'
    samples = []
    with open(log_path) as f:
        for line in f:
            samples.append(line)
    return samples

# batch processing generator
def process_driving_data(samples, batch_size):
    imgs_path = data_path + '/IMG/'
    num_samples = len(samples)

    while 1:
        samples = shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            imgs = []
            measurements = []

            batch_samples = samples[offset:offset+batch_size]

            for batch_sample in batch_samples:
                center_img_path, left_img_path, right_img_path, steering, throttle, brake, speed = batch_sample.strip().split(',')
                center_img_name = center_img_path.split('/')[-1]
                center_img_updated_path = imgs_path + center_img_name
                img = misc.imread(center_img_updated_path)
                img_steering = float(steering.strip())
                imgs.append(img)
                measurements.append(img_steering)
                img_flipped = np.fliplr(img)
                img_flipped_steering = -1 * img_steering
                imgs.append(img_flipped)
                measurements.append(img_flipped_steering)

            imgs_array = np.array(imgs)
            measurements_array = np.array(measurements)
            yield shuffle(imgs_array, measurements_array)

# parsing the arguments
parser = argparse.ArgumentParser(description='Train a network for udacity Behavioural Cloning Project')
parser.add_argument('--data_path', dest='data_path')
parser.add_argument('--model_name', dest='model_name')
parser.add_argument('--nb_epoch', dest='nb_epoch', type=float)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)


args = parser.parse_args()
data_path = args.data_path
nb_epoch = args.nb_epoch
batch_size = args.batch_size
model_name = args.model_name

if data_path[-1] == '/':
    data_path = data_path[:-1]

# reading the data
samples = read_data_lines(data_path)

# 80%-20% train validation split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# train and validation generators
train_generator = process_driving_data(train_samples, batch_size=batch_size)
validation_generator = process_driving_data(validation_samples, batch_size=batch_size)

# building the model
model = Sequential()
# cropping layer that crops 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# lambda layer for normalization of the pixel values
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# 5x5 convolutioal layer with 24 filters followed by RELU activation
model.add(Convolution2D(24, 5, 5, activation='relu'))
# Max-pooling layer with 2x2 pooling filter size
model.add(MaxPooling2D(pool_size=(2, 2)))
# 5x5 convolutioal layer with 36 filters followed by RELU activation
model.add(Convolution2D(36, 5, 5, activation='relu'))
# Max-pooling layer with 2x2 pooling filter size
model.add(MaxPooling2D(pool_size=(2, 2)))
# 5x5 convolutioal layer with 48 filters followed by RELU activation
model.add(Convolution2D(48, 5, 5, activation='relu'))
# Max-pooling layer with 2x2 pooling filter size
model.add(MaxPooling2D(pool_size=(2, 2)))
# 5x5 convolutioal layer with 64 filters followed by RELU activation
model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
# Flatten all output filter maps
model.add(Flatten())
# Fully connected layer with 1064 units
model.add(Dense(1064))
# Dropout layer with 50% dropout rate
model.add(Dropout(0.5))
# Fully connected layer with 100 units
model.add(Dense(100))
# Dropout layer with 30% dropout rate
model.add(Dropout(0.3))
# Fully connected layer with 50 units
model.add(Dense(50))
# Fully connected layer with 10 units
model.add(Dense(10))
# Final regression unit
model.add(Dense(1))
# adam optimizer to minimize mean square error
model.compile(loss='mse', optimizer='adam')
# Training the model
model_history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
 validation_data=validation_generator, nb_val_samples=len(validation_samples),
  nb_epoch=nb_epoch, verbose=1)
# saving the model
model.save(model_name)

# plot the training and validation loss for each epoch
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(model_name[:-3] + '_loss.jpg')

# Exit the program
exit()

