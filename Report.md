# Deep Reinforcement Learning Project #1 - Report
Here below are reported the main characteristics of the Reinforcement Learning algorithm used to solve the DRLND <em>Navigation</em> environment.

## Learning Algorithm
The applied algorithm is the Deep Q-Network (DQN). 
Four neural networks are used; Actor and Critic Networks are backed by their corresponding target network and soft-updated by `tau`.
Values of `tau`small enought, ensure stability during the training process.
The Ornstein-Uhlenbeck process is used to introduce noise and ensure exploration during training.
More details of the DDPG can be found [here](https://arxiv.org/abs/1509.02971).
Few variations of the DQN implementation are included in the agent:
- Dual Deep Q-Learning: where, during the learning step, the agent 
- Duelling networks: where the agent use 

## Model and Parameters
The file `model.py' holds the classes of the Neural Network for the generic DQN and the Dueling DQN.



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

Running the trained model, the agent achieves easly 20 points in one single episode. 

## ToDo list
Much more stuff can be done to around the project.

### Training Algorithm improvements
This Deep Q-Learning algorithm can be still improved with proved extensions.
Here the planned implementations:
1.Prioritized replay
2.Distributional DQN

### Deep Analysis of the Paramenters and Architecture
Wide analysis of the impact of the training parameters and model architecture on the training performance. A specific Jupyter Notebook will be set up for the purpose.

### Step to Pixel Based State Space
Use the *Navigation_Pixels.ipynb* and adapt the agent code to solve the banana collection enviroment using raw pixels. That will require mainly the modification of the `model.py` to include convolutional layers.
