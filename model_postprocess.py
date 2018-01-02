### Initializations

# Imports
import csv
import cv2
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
from keras.utils.visualize_util import plot
from keras.backend import clear_session

### Model post processing

# check that model Keras version is same as local Keras version
f = h5py.File('model.h5', mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
	print('You are using Keras version ', keras_version,
		  ', but the model was built using ', model_version)

model = load_model('model.h5')

# Open training history
with open('train_hist.p', 'rb') as pickle_file:
	history = pickle.load(pickle_file)

print('History opened in train_hist.p')

# Visualize model and save it to disk
model.summary()
plot(model, to_file='img/model.png', show_shapes=True, show_layer_names=False)
print('Saved model visualization at img/model.png')

# Print the keys contained in the history object
print(history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model: Mean Squared Error Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.savefig('img/lossPlot.png')

# Clear Keras Session
clear_session()
