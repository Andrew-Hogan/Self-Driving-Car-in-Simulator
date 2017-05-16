# Self-Driving-Car-in-Simulator


Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.

Cleanmodel is the model definition for my convolutional neural network used in driving a simulated vehicle autonomously.

Anglecalcfar was a tool used in determining the relationship between the rate of change in offset cameras, the center camera steering angle, and the distance from the vehicle to its goal. This was important for data augmentation.

Drive utilizes the model generated from Cleanmodel and transmits commands to the simulator.

Model is for reference on what Cleanmodel looked like when it was graded and approved.
