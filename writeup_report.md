#**Behavioral Cloning** 

##Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The data import process is followed by the preprocessing function and generator; which is followed by the model definition and execution function - and finally a model accuracy visualization through the history object.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with two 5x5 filter sizes followed by two 3x3 filter sizes and depths between 24 and 64 (model.py lines 95-121) 

The model includes RELU layers to introduce nonlinearity (code line 101 and following each conv/fc layer), and the data is normalized in the model using a Keras lambda layer (code line 98). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 35).

Half the data was randomly flipped (code line 54) to prevent a tendency for one direction or another.

60% of data (along with the corresponding side cameras) with a steering angle of less than .2 radians was removed due to a prevelance of these steering angles. (code line 10-32)

The left and right camera angles were imported with a dynamic, rather than static, offset. This resulted in incredibly strong performances on only one run of the track. While Udacity's classroom, as well as every student I know of, used a static offset (around .1 radians) to the left or right; I found that by increasing the offset more for the camera angle whose absolutely value needed to be increased (such that if a camera usually was offset by negative .1 and the center angle was already negative, the offset would be even greater than .1) I was getting better results with far less training data than classmates. By fine tuning the paramaters, and going through a lot of trigonometry, I found that keeping the angle whose absolute value needed to be decreased as a static offset and increasing the other by the angle multiplied by the angle over the maximum possible angle added to the original offset, I was getting a maximum possible offset of 2 times the angle plus the original offset. This is due to some basic trigonometry. If the goal that the steering angles are directing the car to is a static distance from the car, then the offsets for the two side cameras will be the same no matter where that goal is in relation to those angles. But if the goal for the car is much closer to the car - as it would be in a sharp turn - then the angle for the side camera whose absolute value needs to be increased ends up being double the center angle plus the original offset, which is determined by the distance between the cameras and the original distance to the hypothetical goal for the car. .1 was the assumption I made on the original offset as the car drove well when driving straight with that offset, but suffered on turns. The offset process can be found in code lines 38-57.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
