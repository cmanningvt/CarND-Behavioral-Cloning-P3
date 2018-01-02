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
from keras.backend import clear_session
#from keras.utils.visualize_util import plot

# Initialize variables
lines = []

# Initialize hyper parameters
batch_size = 256
epochs = 40

### Function Definitions

# Define generator function
def generator(samples, batch_size=32):
	num_samples = len(samples)
	side_camera_angle_offset = 0.1
	
	start = 0
	stop = start + batch_size
	
	sklearn.utils.shuffle(samples)
	
	while True:
		
		batch_samples = samples[start:stop]
		images = []
		measurements = []
		
		for batch_sample in batch_samples:
			# Read in path for center image and adapt path to AWS directory
			source_path_center = batch_sample[0]
			filename_center = source_path_center.split('/')[-1]
			current_path_center = 'data/IMG/' + filename_center
			image_center = cv2.imread(current_path_center)
			image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
			
			# Read in path for left image
			source_path_left = batch_sample[1]
			filename_left = source_path_left.split('/')[-1]
			current_path_left = 'data/IMG/' + filename_left
			image_left = cv2.imread(current_path_left)
			image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
			
			# Read in path for right image
			source_path_right = batch_sample[2]
			filename_right = source_path_right.split('/')[-1]
			current_path_right = 'data/IMG/' + filename_right
			image_right = cv2.imread(current_path_right)
			image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
			
			# Augment data with reversed image
			image_center_flipped = np.fliplr(image_center)
			image_left_flipped = np.fliplr(image_left)
			image_right_flipped = np.fliplr(image_right)

			# Add images to list
			images.append(image_center)
			images.append(image_center_flipped)
			images.append(image_left)
			images.append(image_left_flipped)
			images.append(image_right)
			images.append(image_right_flipped)
			
			# Read steering angles and add to list
			measurement = float(batch_sample[3])
			measurements.append(measurement)
			measurements.append(measurement * -1)
			measurements.append((measurement + side_camera_angle_offset))
			measurements.append((measurement + side_camera_angle_offset) * -1)
			measurements.append((measurement - side_camera_angle_offset))
			measurements.append((measurement - side_camera_angle_offset) * -1)

			# Convert to np array types
			X = np.array(images)
			y = np.array(measurements)
			
			start += batch_size
			stop += batch_size
			
			if start >= num_samples:
				start = 0
				stop = batch_size
				
			sklearn.utils.shuffle(X, y)
			
			yield (X, y)
		
### Data import

# Open CSV file
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Remove header row
del lines[0]

# Split training set
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# Compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=batch_size)
validation_generator = generator(validation_lines, batch_size=batch_size)

### NN Model

# Model definition
model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (dimensions)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))

# Cropping images
model.add(Cropping2D(cropping = ((70,25),(0,0))))

# Nvidia model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Model optimization and training
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])
#history = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = epochs, batch_size = batch_size)
h = model.fit_generator(train_generator, samples_per_epoch=len(train_lines)*6, validation_data=validation_generator, nb_val_samples=len(validation_lines)*6, nb_epoch=epochs)
history = h.history

### Model post processing

# Save model architecture to model.json, model weights to model.h5
json_data = model.to_json()
with open('model.json', 'w') as f:
	f.write(json_data)
model.save('model.h5')

# Save training history
with open('train_hist.p', 'wb') as f:
	pickle.dump(history, f)

print('Model saved in model.json/h5, history saved in train_hist.p')

# Clear Keras Session
clear_session()