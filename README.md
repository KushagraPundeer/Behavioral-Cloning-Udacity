# Udacity-Behavioral-Cloning-

The goals / steps of this project are the following:

• Use the simulator to collect data on Track – 1, try keeping the vehicle
centered on the lap.

• Build, a convolution neural network in Keras that predicts steering angles
from images

• Train and validate the model with a training and validation set

• Test that the model successfully drives around track one without leaving
the road

• Summarize the results with a written report

My project includes the following files:
• model.py containing the script to create and train the model

• drive.py for driving the car in autonomous mode

• model.h5 containing a trained convolution neural network

• video.mp4, video for the track 1 first lap.

# Model Architecture and Training Strategy. 

Model is based on the Comma.ai’s model. The model contains dropout layers to reduce overfitting.
The model used an adam optimizer, lr = 0.0001 to automatically tune the learning
rate to the desired level.

Training data was chosen to keep the vehicle driving on the road. I used the
Udacity data collected along with additional data collected for 3 laps on track 1. I
added 3 laps of data taken in opposite direction on track 1 to the above data to
train the network

# Solution Design Approach
I tried the simple model, Nvidia model, modifed versions of the nvdia model and
comma.ai’s model. Simple model along with some variants did well on straighter
sections of the track 1 but went off track on sharp turns. I ended up choosing
comma.ai’s model as it performed well.

Choosing ELU for activation and using dropout helped with the performance.
In order to gauge how well the model was working, I split my image and steering
angle data into a training and validation set. I removed 65% data with steering
angle zero for training data. I checked the distribution and got a nice bell shaped
curve.

# Final Model Architecture: Modified Comma.ai’s model
Convolution Layer 1:
Filters: 16 | Kernel: 8 x 8 | Stride: 4 x 4 | Padding: Same | Activation: ELU

Convolution Layer 2:
Filters: 32 | Kernel: 5 x 5 | Stride: 2 x 2 | Padding: Same | Activation: ELU

Convolution Layer 3:
Filters: 64 | Kernel: 5 x 5 | Stride: 2 x 2 | Padding: SAME | Activation: ELU

Flatten layer
Fully Connected Layer 1

Neurons: 512 | Dropout: 0.5 | Activation: ELU

Fully Connected Layer 2:
Neurons: 50 | Activation: ELU


# Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded three laps on track one using
center lane driving. I then collected 3 additional laps in the reverse direction.
To augment the data sat, I also flipped images and angles for the center camera. 

I normalized the training data and cropped the images and hence makingimage
input size for training model to be (80,320,3). I used this training data for training the model. The validation set helped
determine if the model was over or under fitting. 

I found 20 to be the ideal no’ of epoch.
Loss: 0.0146 (after 20 epochs)
val_loss: 0.0239.

I used keras Model.fit_generator along with a custom built generator to generate
training data. The data was filpped for the center camera along with adding
steering correction to left and right camera images if there was augmentation.
The tested the model in the simulator for two laps and performed well and stayed
on the track. I used the video.py script to create a mp4 videos of the images
collected on the simulator for track 1. I tried the model on track 2 for self
reference. The model worked well on most of the track 2 but had a slight left
turning tendency which caused issues. I think it can be solved by collecting
additional data on track 2 and normalizing the data more.
