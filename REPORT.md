For this project we have used the [DDPG Algorithm](https://arxiv.org/abs/1509.02971) in order to train the Agent. 

We used the environment provided by Udacity choosing the case of 20 agents ('Reacher_Linux_NoVis/Reacher.x86_64')

## Algorithm and Model Architecture

The architecture is comprised of two neural networks one for the Actor and one for the Critic as described below:

### Actor NN

* hidden layer: (input, 256) - ReLU
* hidden layer: (256, 128) - ReLU
* output layer: (128, 4) - tanH

### Critic NN

* hidden layer: (input, 256) - ReLU
* hidden layer: (256 + action size, 128) - ReLU
* output layer: (128, 1) - Linear

Using the Actor-Critic paradigm the training loop is composed out of two steps: acting and learning. 
In the acting step, the agent passes the state vector through the Actor network and takes the action which is the output of the network.
In the learning step, the Critic network is used as a feedback to the Actor network to change its weights such that the estimated value of the input state is maximized.

We also use Replay Buffer.  

## Hyperparameters
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor
* LR_CRITIC = 1e-4        # learning rate of the critic
* WEIGHT_DECAY = 0      # L2 weight decay

## Results
Using the aforementioned architecture and hyperparameters we were able to solve the environment in 108 episodes scoring above 30.0. 

We can see the performance of the model in the following diagram: 
![alt text](https://github.com/Strihias/Deep-Reinforcement-Learning---Reacher-Project/blob/main/diagram.jpg "Performance Diagram")

## Future Work
I aim to experiment more on this project by implementing other algorithms like the [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/forum?id=SyZipzbCb)
