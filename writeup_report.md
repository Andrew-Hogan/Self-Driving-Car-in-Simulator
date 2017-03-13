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

[image1]: ./examples/architecture.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Center Lane"
[image3]: ./examples/center_recover.jpg "Recovery Image"
[image4]: ./examples/center_recover2.jpg "Recovery Image"
[image5]: ./examples/center_recover3.jpg "Recovery Image"

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
* video.mp4 which showcases the results

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

I'll skip straight to what makes this implementation different from everyone else's, and what has made the most difference with the least amount of data. The left and right camera angles were imported with a dynamic, rather than static, offset. This resulted in incredibly strong performances on only one run of the track. While Udacity's classroom, as well as every student I know of, used a static offset (around .1 radians) to the left or right; I found that by increasing the offset more for the camera angle whose absolutely value needed to be increased (such that if a camera usually was offset by negative .1 and the center angle was already negative, the offset would be even greater than .1) I was getting better results with far less training data than classmates. By fine tuning the paramaters, and going through a lot of trigonometry, I found that modifying the angle whose absolute value needed to be decreased as .928 multiplied by the center angle plus the additional offset and adjusting the other angle by the angle multiplied by ~1.78 added to the original offset, I was getting the best results possibly given minimum data. This is due to some basic trigonometry, and makes major simplifications to the dynamic offset and assumptions about a change in the center angle reflecting a change in the distance to the goal. If the goal that the steering angles are directing the car to is a static distance from the car, then the offsets for the two side cameras will be the same no matter where that goal is in relation to those angles. But if the goal for the car is much closer to the car - as it would be in a sharp turn - then the angle for the side camera whose absolute value needs to be increased ends up being almost double the center angle plus the original offset, which is determined by the distance between the cameras and the original distance to the hypothetical goal for the car. My model assumed an arbitrary unit of ~1 between each camera with a nonarbitrary unit of ~9 to the goal while driving straight, which resulted in a baseline offset of ~0.1 radians. The offset process can be found in code lines 38-57. To see how I calculated the increase for the farthest camera from the goal (commented with proofs, assumptions, and variables) there is an anglecalcfar.py file on my github [github](https://github.com/Andrew-Hogan/Self-Driving-Car-in-Simulator) which produces a graph - from which I derived the equation, which only works for angles relevant to this simulation (0 - 0.5 radians, the max steering angle). I would love to get deeper into the relationship between the different offsets to create a cohesive formula for calculating relative change given a center angle and a distance, but my formula is as effective and accurate as it needs to be for this scope.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 35).

Half the data was randomly flipped (code line 54) to prevent a tendency for one direction or another.

60% of data (along with the corresponding side cameras) with a steering angle of less than .2 radians was removed due to a prevelance of these steering angles. (code line 10-32)

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used one lap of driving from the center of the road on each track, which allowed it to follow the center without falling off - and then used an additional recovery lap. However, these were before implementing the optimized camera offset and I was able to get it to stay on the road using only one lap without recovery afterwards.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement a successful architecture I had worked on before; which had two 3x3 convolutions followed by 5x5 and fully connected layers, and then test it. It worked incredibly well on the first track on the first go, but suffered on the challenge track. I figured this indicated the model was not complex enough for generalization between the tracks.

My second step was to use a convolution neural network model similar to the NVIDIA self driving car architecture. I thought this model might be appropriate because it is built specifically for self driving cars. I simplified down the depth and cut some layers, as a much simpler but similar model (my previous one) was working already.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training and validation set. This pretty much never changed regardless of whether or not the car was driving well, but anything below .1 and anything above .3 on either training or valdiation generally meant the model would perform poorly.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle did not ever fall off the track on track one. Track two required a rework of the architecture, and to get it to drive in a way that was not unsafe (as opposed to just not falling off) required fine tuning of the offcenter camera angle import process.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and without drifting from the center line in a way which would alarm passengers. In fact, it drove far better than I did with an xbox controller on either map.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
Input: 40, 80, 3
Cropping2D: (10,5), (0,0)
Lambda (Normalization)
Convolution: 5x5x24
Relu
Convolution: 5x5x36
Relu
MaxPooling: (2,2)
Convolution: 3x3x48
Relu
Convolution: 3x3x64
Relu
Flatten - Fully Connected Layer: 1164
Relu
Fully Connected: 100
Relu
Fully Connected: 50
Relu
Fully Connected: 10
Relu
Fully Connected: 1
All padding was valid.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the side of the lane in case it drifted off center. These images show what a recovery looks like starting from the right:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I cut 60% of data points with a center steering angle of less than .2 radians, including the associated side cameras and angles.

After the collection and import process, I had 4565 data points. I then preprocessed this data by using the aforementioned offset process to create three times as much data. I randomly flipped half of the images and angles thinking that this would prevent the neural network from prefering one side or another without needing to collect additional data. I resized the image to an 80 by 40 numpy array, which was later zero centered and cropped by 15 pixels vertically with a keras lambda and cropping layer. This allowed the network to ignore unneccessary data and sped up training time.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by a change in training and validation loss less than 10% of the previous epoch's. I used an adam optimizer so that manually training the learning rate wasn't necessary.
