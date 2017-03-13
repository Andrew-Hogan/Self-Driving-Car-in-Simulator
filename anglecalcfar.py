import math
import numpy as np

pi = math.pi ##pi
counter = 0 ##tracks where program is in calculating the changes in side camera a
maxruns = 50 ##number of times to calculate
increment = (.5)/(maxruns) ##angle increment between runs
b = increment ##hyperparameter #1 which is incremented by increment, determines center steering angle
n =  .1 + (b * b) / (.45) ##hyperparameter #2 determining the length of the goal to the object which increments closer as steering angle increases (-is original side angle given 0 steering angle)
x = 1 ##hyperparameter #3 determining the distance between cameras

def findchange(n, b, x):
    ##n = hyperparameter #1 determining original angle offset
    ##b = hyperparameter #2 determining offset for 90 degree angle in second triangle
    ##x = hyperparameter #3 determining the distance between cameras
    pi = math.pi
    ab = (.5 * pi) - n
    a = math.tan(ab) * x ## tan = opp/adj
    bn = math.sqrt(math.pow(a, 2) + math.pow(x, 2) - (2 * a * x * math.cos((.5 * pi) + b))) ##law of cosines
    ac = math.asin((a * math.sin((.5 * pi) + b)) / bn) ##law of sines
    if ac > (.5 * pi):
        ac = (.5 * pi) - (ac - (.5 * pi))
    #ad = ac - ab
    ac = (.5 * pi) - ac
    print (b, ac)
    return ac

counter = 0 ##tracks where program is in calculating the changes in side camera a
maxruns = 50 ##number of times to calculate
angles = np.zeros([50, 2]) ##array of results

while counter < maxruns: #loop for storing
    angles[counter, 1] = findchange(n, b, x)
    angles[counter, 0] = b
    b += increment
    n =  .1 + (b * b) / (.45)
    counter += 1

import matplotlib.pyplot as plt
plt.plot([angles[:,0]], [angles[:,1]], 'ro')
plt.show()
