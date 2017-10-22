# Using Deep Q-Network to Learn How To Play Flappy Bird

<img src="./images/flappy_bird_demp.gif" width="250">

## Overview
This project applies the Deep Q Learning algorithm and shows that this learning algorithm can be further generalized to the game Flappy Bird.

## Installation Dependencies:
* Python==3.5
* TensorFlow==1.3
* pygame
* OpenCV-Python

## How to play the game:
```
git clone https://github.com/
cd DeepLearningFlappyBird
python deep_q_network.py
```

## How to train the model by yourself:
Set TRAINING_MODE as True on Line 10 in deep_q_network.py

#### Network Architecture
I first preprocessed the game screens with following steps:

1. Convert image to grayscale
2. Resize image to 80x80
3. Stack last 4 frames to produce an 80x80x4 input array for network

The architecture of the network is shown in the figure below. The first layer convolves the input image with an 8x8x4x32 kernel at a stride size of 4. The output is then put through a 2x2 max pooling layer. The second layer convolves with a 4x4x32x64 kernel at a stride of 2. We then max pool again. The third layer convolves with a 3x3x64x64 kernel at a stride of 1. We then max pool one more time. The last hidden layer consists of 256 fully connected ReLU nodes.

<img src="./images/network.png">

The final output layer has the same dimensionality as the number of valid actions which can be performed in the game, where the 0th index always corresponds to doing nothing. The values at this output layer represent the Q function given the input state for each valid action. At each time step, the network performs whichever action corresponds to the highest Q value using an ϵ greedy policy.


#### Training
At first, I initialize all weight matrices randomly using a normal distribution with a standard deviation of 0.01, then set the replay memory with a max size of 500,00 experiences.

I start training by choosing actions uniformly at random for the first 10,000 time steps, without updating the network weights. This allows the system to populate the replay memory before training begins.

Note that unlike [1], which initialize ϵ = 1, I linearly anneal ϵ from 0.1 to 0.0001 over the course of the next 3000,000 frames. The reason why I set it this way is that agent can choose an action every 0.03s (FPS=30) in our game, high ϵ will make it **flap** too much and thus keeps itself at the top of the game screen and finally bump the pipe in a clumsy way. This condition will make Q function converge relatively slow since it only start to look other conditions when ϵ is low.
However, in other games, initialize ϵ to 1 is more reasonable.

During training time, at each time step, the network samples minibatches of size 32 from the replay memory to train on, and performs a gradient step on the loss function described above using the Adam optimization algorithm with a learning rate of 0.000001. After annealing finishes, the network continues to train indefinitely, with ϵ fixed at 0.0001.

## Disclaimer
This work is adopted from Lin's[https://github.com/yenchenlin/DeepLearningFlappyBird] work.

