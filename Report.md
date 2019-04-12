# Deep Reinforcement Learning Project #1 - Report
Here below are reported the main characteristics of the Reinforcement Learning algorithm used to solve the DRLND <em>Navigation</em> environment.

## Learning Algorithm
The applied algorithm is the Deep Q-Network (DQN). 
Considering the nature of the action/state space the selected RL Algorithm was a Deep Deterministic Policy Gradient (DDPG).
Four neural networks are used; Actor and Critic Networks are backed by their corresponding target network and soft-updated by `tau`.
Values of `tau`small enought, ensure stability during the training process.
The Ornstein-Uhlenbeck process is used to introduce noise and ensure exploration during training.
More details of the DDPG can be found [here](https://arxiv.org/abs/1509.02971).

## Model and Parameters
The models of the Critic and Actor are DNN of fully connected layers.
The Actor holds two hidden layers of 256 nodes each.
The Critic holds an hidden layer of 256 + action space and a second hidden layer of 128 nodes. 
The two agent are sharing the two networks during training.

Here the summary of parameters used:

`lrate_critic=1e-3,
lrate_actor=1e-4,
tau=0.01,
gamma=0.99,`

Replay Buffer
`size = 10000,
batch_size = 256`

Ornstein-Uhlenbeck noise
`exploration_mu=0.0,
exploration_theta=0.25,
exploration_sigma=0.30,
noise_decay=0.99995`

## Results

The trend below shows the score avarege on 100 episode achieved by the agent across the learning process.
The trend compares the DQN Vanilla with the improved methods implemented. a second trend shows the variance of the results. 
![Results](results/Training_Process.png)

All the methods overcome the target of average reward (over 100 episodes) of +13 and they stabilize around +16. This limit is due to the duration of the episode and the variability of the result is related to the stocastic distruburion of the bananas and the residual noise necessary to commit to exploration during training.

Running the trained model, the agent achieves easely 20 points within one single episode. 
