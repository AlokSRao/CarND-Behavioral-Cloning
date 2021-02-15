# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/lossvsepoch.png "Loss Vs Epoch"
[image2]: ./examples/architecture.png "Neural Net Architecture"
[image3]: ./examples/NVIDIA.png "NVIDIA end to end architecture"
[image4]: ./examples/layersizes.png "output sizes"
[image5]: ./examples/left.jpg "output sizes"
[image7]: ./examples/right.jpg "output sizes"
[image6]: ./examples/center.jpg "output sizes"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For my model architecture, I implemented the NVidia [End to End Learning for Self-Driving Cars.](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

It consists of 5 convolution layers -> flattening layer -> Fully connected layer -> Dropout -> Fully connected layer -> Dropout Fully connected layer -> Fully connected layer.

I used the mean squared error to train it, using the ADAM optimiser.

The data was preprocessed normalising it and cropping it so that only relevent road section remains
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 130). 

I used the sklearn train_test_split library to split the driving data into training and validation data(20% of the data)

I then used the keras fit_generator function to train it on train test and validate on validation set

#### 3. Model parameter tuning

The model used an adam optimizer, with learning rate of 0.0001. I reduced the learning rate to reduce overfittinh

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left lane driving and right lane driving with correction factor to generate the training and validation set

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I leveraged existing NVIDIA research to design my neural net.

It consists of 5 convolution layers -> flattening layer -> Fully connected layer -> Dropout -> Fully connected layer -> Dropout Fully connected layer -> Fully connected layer.

I used the mean squared error to train it, using the ADAM optimiser.

The data was preprocessed normalising it and cropping it so that only relevent road section remains

The dropout probability was 0.5.

I also trained it on a batch size of 256, running it for 5 epochs

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
While researching behaviour cloning, I came across the NVIDIA end to end model paper, and decided to implement it, as I was very satisifed by the results they had achieved. The [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) describes their setup, which is nearly identical to our setup of one center camera, one left camera, and one right camera. THe following is an image of the network, directly from the paper mentioned above:
![alt text][image3]

However, probably due to size of data sets, direct implementation of above architecture resulted in overfittin, and did not produce good results. Hence, I added dropout layers alternatively, which improved the results.
The final model architecture consisted of a convolution neural network with the following layers and layer sizes. 

![alt text][image2]
![alt text][image4]

NOTE: Data was normalised and cropped before passing through the model

#### 3. Creation of the Training Set & Training Process

The simulator set up provided us with three image types - Center camera, left camera, right camera.
![alt text][image5] ![alt text][image6] ![alt text][image7]

THe simulator also provided corresponding angles, associated with the images. Ideally, only the center images can be passed to the model. However, to augement the data, I did the following steps:
1.Used a correction factor to use the left data, so that the model could handle going to the left, same with right image
2.Flipped the image, and flipped sign of steering angle, to provide more variance

Using the methods mentioned above, I was able to produce the following results:

![alt text][image1]