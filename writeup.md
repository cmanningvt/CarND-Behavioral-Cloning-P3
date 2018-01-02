# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/lossPlot.png "Loss Plot"
[image3]: ./img/center.jpg "Center Image"
[image4]: ./img/right.jpg "Recovery Image"
[image5]: ./img/left.jpg "Recovery Image"
[image6]: ./img/center_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  Many decisions were made with the help of the [Behavioral Cloning Cheat Sheet](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb)

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* track1.mp4 with a video of my recorded autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of using the [nVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) mentioned in the course contents. This has convolutional layers with 3x3 and 5x5 fileters with depths between 24 and 64 (model.py lines 121-139).
I added a cropping layer in the beginning to make sure that any cropping effects were included when running the trained model with the similator. Doing cropping during the prepossessing steps would have caused issues when running with the simulator.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 124). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 142).

#### 4. Appropriate training data

I had a lot of issues getting driving the simulator manually. I felt like I could never keep the vehicle on the road for more than 5 seconds. Due to this, I relied solely on the sample data provided with the project template. This lacked recovery data, but I felt like I was able to train the model ok using the left and right camera images for recovery efforts.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the nVIDIA network. I thought this model might be appropriate because it was designed for calculating steering angle from image data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. So I reduced the number of epochs to find a good point of stability.

![alt text][image2]

I also added a cropping step to the model so that training, validation, and test (simulator) data was all evaluated equally. 

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I was unable to collect data reliably by myself so I used the sample data from the project template. Here is an example image of center lane driving:

![alt text][image3]

I also used images from the left side and right sides of the road back to center so that the vehicle would learn to recover from being off center of the road.

![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would generalize the model since most of the track involves left hand turns. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image6]

I then preprocessed this data by normalizing the data and cropping it to remove sky and vehicle parts.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 40. I used an adam optimizer so that manually training the learning rate wasn't necessary.
