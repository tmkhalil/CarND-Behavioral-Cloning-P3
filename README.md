# **Behavioral Cloning** 

## Project writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example0.jpg "Example Image 1"
[image2]: ./examples/example1.jpg "Example Image 2"
[image3]: ./examples/example2.jpg "Example Image 3"
[image4]: ./examples/example3.jpg "Example Image 4"
[image5]: ./examples/self_steering_model0.jpg "Validation Performance over epochs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* self_steering_model0.h5 containing a trained convolution neural network 
* model0_vid.mp4 a video for the car in the autonomous mode.
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python self_steering_model0.h5
```

#### 3. Submission code is usable and readable

The self_steering_model0.h5 file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is Based on Convolutional Neural Networks and uses 5x5 and 3x3 kernel sizses. The model uses normalization, croping and dropout layers.
More architichural details are provided in the model architecture section.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, I used Max-Pooling layers after my convolutioal layers and Dropout layers after some of the fully connected layers. The Data also was augmented with flipped images.
The details are provided in the model architecture section.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123).

### Model Architecture and Training Strategy

#### 1. Creation of the Training Set & Training Process

For the data collection part, I followed all the helpful hints of the course instructors by:
- Collecting as much data as I could by trying to drive properly in the middle of the road.
- Then, collect more data to capture the fact that we want the car to be as close to the middle of the road as possible by trying to record as many frames as possible of the car while moving from the sides of the road to the center.
- Doing the same in the first two steps but driving in the reverse direction to help the model to generalize.
- Data augmentation by introducing new examples by flipping the original ones and reverse the steering angle sign.
- Only center images were filtered and used for training and validation.
- As a result of all the previous steps, I collected a number of 37622 training examples before data augmentation and the double after data augmentation (75244 examples).

Here are some examples of training the car to drive in the middle

![alt text][image1]
![alt text][image2]

And here some examples where the car is starting to recover when it's not in the middle of the road

![alt text][image3]
![alt text][image4]

#### 2. Solution Design Approach and Final Model Architecture

To validate the performance, I splitted the dataset into train/validation based on 80% to 20% split ratio.

After watching the Video lectures of the project, I started with what worked best which was NVIDIA architecture. NVIDIA architecture includes normalization layer but not cropping one which proved to be useful in the video lectures, so I added it.

When I started with the basic [NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and tested it in the simulation mode, It seemed that the model is a bit overfitting.

Then I modified the model by addining dropout and Max-Pooling layers and by choosing the best number of epochs to avoid overfitting.

I trained the model and given the data provided in the following chart. I chose 5 epochs as a good number of epochs where the performace on the validation set was stable and was consistently decreasing until that point, after that it started to fluctuate from one epoch to another.

![alt text][image5]

Using the output model, the car was very stable and successfully ran into the first track as shown in the provided video.

The detailed model architecture was as follows (from line 86 to 123 in model.py file):

- Cropping layer to make use of only useful information by cropping 70 pixels from the top of the image and 25 from the bottom.

- Lambda layer for normalization of the pixel values

- 5x5 convolutioal layer with 24 filters followed by RELU activation

- Max-pooling layer with 2x2 pooling filter size

- 5x5 convolutioal layer with 36 filters followed by RELU activation

- Max-pooling layer with 2x2 pooling filter size

- 5x5 convolutioal layer with 48 filters followed by RELU activation

- Max-pooling layer with 2x2 pooling filter size

- 5x5 convolutioal layer with 64 filters followed by RELU activation

- Flatten all output filter maps

- Fully connected layer with 1064 units

- Dropout layer with 50% dropout rate

- Fully connected layer with 100 units

- Dropout layer with 30% dropout rate

- Fully connected layer with 50 units

- Fully connected layer with 10 units

- Final regression unit

The model aimed to minimize the Mean Square Error (MSE) using adam optimizer.