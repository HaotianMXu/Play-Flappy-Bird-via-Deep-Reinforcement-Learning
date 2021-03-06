# Using Deep Q Network to Learn How To Play Flappy Bird

## Overview
This project applies the Deep Q Learning algorithm to teach your computer how to play flappy bird:)

<img src="./images/flappy_bird_demp.gif" width="200">

## Requirements:
* [Python==3](https://www.anaconda.com/download/)
* [TensorFlow](https://www.tensorflow.org/install/)
* [Pygame](http://www.pygame.org/wiki/GettingStarted#Pygame)
* [OpenCV-Python](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)


## How to play the game:
```
git clone https://github.com/HaotianMXu/Play-Flappy-Bird-via-Deep-Reinforcement-Learning.git
cd DeepLearningFlappyBird
python deep_q_network.py
```
## How to train the model by yourself:
Set TRAINING_MODE as True at Line 10 in deep_q_network.py

## Details
#### Preprocessing
1. Convert each frame to grayscale
2. Resize to 80x80
3. Stack every four frames to produce an 80x80x4 input array for network
#### Network Architecture
This network contains three convolution and max pooling layers and two fully connected layers:

The first convolution layer applies 32 8x8 kernels with stride=4. The output is then put through a 2x2 max pooling layer. The second convolution layer carries 64 4x4 kernels with stride=2. A 2x2 max pooling layer is followed. The third convolution layer convolves with 64 3x3 kernels with stride=1. Then the output is passed through another 2x2 max pooling layer. At last, two fully connected layers with 256 hidden nodes are applied before making the final decision.

The final output layer has the same dimensionality as the number of valid actions which can be performed in the game, where the first node corresponds to doing nothing and the second corresponds to flying up in our case. This output layer represents Q function given the input state for each valid action. At each frame, the network chooses action with the highest Q value. ϵ greedy policy is utilized to balance exploration and exploitation.
#### Parameter Setup
For convolution filters, I initialize all weights randomly using a normal distribution with standard deviation=0.01. The max size of replay memory is set as 50k, while the memory is generated by choosing actions uniformly for the first 10k time steps without updating the network weights. This allows the system to populate the replay memory before training begins.

For the ϵ greedy algorithm, I linearly anneal ϵ from 0.1 to 0.0001 during the next 3000k frames which means that the model has probability 0.1 to select a random action at the beginning and probability 0.0001 at the end. It is reasonable to start from a large ϵ because the model is not reliable at the beginning and needs more exploration.
#### Training stage
At each time step, the network samples minibatches from the replay memory for training, and updates weights using the Adam optimization algorithm with a learning rate of 1e-6. After annealing finishes, the network continues to train with ϵ fixed at 0.0001.

## Acknowledgement
This work is adapted from [Lin's code](https://github.com/yenchenlin/DeepLearningFlappyBird).

