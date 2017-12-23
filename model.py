# Imports for data import and analysis
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D

# Initialize variables
lines = []

# Initialize hyper parameters
batch_size = 128
epochs = 10

# Define generator function
def generator(samples, batch_size=32):
	num_samples = len(samples)
	side_camera_angle_offset = 0.1
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

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

			# trim image to only see section with road
			X = np.array(images)
			y = np.array(measurements)
			dimensions = X_train[0].shape
			yield sklearn.utils.shuffle(X, y)

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

# Model definition
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (dimensions)))

# Nvidia model
#model.add(Cropping2D(cropping = ((70,25),(0,0))))
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
history = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = epochs, batch_size = batch_size)
model.fit_generator(train_generator, samples_per_epoch=len(train_lines), validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=epochs, shuffle = True)

# Save model
model.save('model.h5')
