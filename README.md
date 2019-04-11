# Deep Reinforcement Learning Project #1 - Navigation

This repository contains the implementation of a DQN Algorithm to train an Agent to navigate the 'bananas' simulation environments from [Unity ML-Agents.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
This aim to solve the navigation problem proposed in the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program by collecting as much as yellows bananas as possible, avoiding the blue ones.

---

## Installation

To run this code, you will need to download the prebuild Unity enviroment not provided in the repository. You need to select the enviroment for your OS:
* [x Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [x Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [x Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [x Windows (64-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Place the file in the DRL_Project#1-Navigation Folder and unzip.

Beside the Unity enviroment, Python 3.6 must be available with the Unity ML-Agents [(see this link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) installed, and few more packages (see `enviroment.env`).
 
## Environment 

The Agent in the banana world need to collect yellow bananas by moving through the 2D domain. To do so, it has 4 actions available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

The Reward function of the Agent can be resumed by:
- +1 when a yellow banana is collected
- -1 when a blue banana is collected
- 0 in the other cases

The task is episodic, and it is considered solved if the agent can get an average score of +13 over 100 consecutive episodes.
 
## Instructions

Open `Navigation.ipynb` and run the code alongside with the provided instructions.
The notebook is devided in few steps:
1. Starting the Unity enviroment and setup the defaul brain to address one agent.
2. Analisys of the State and Action Spaces provided by the Unity enviroment.
3. Random Agent acting in the enviroment (example of the interaction between an agent and the enviroment).
4. Initial setup of the parameters of the DQN Algorithm to train the agent.
5. DQN Vanilla training.
6. Double DQN training.
7. Dueling Network Architecture for DQN training.
8. Combination of the DQN improvements at points 6 and 7.
9. Results comparison.
10. An example of a trained agent in the bananas collection enviroment.

The file `my_methods.py` provides the implementation of the DQN algorithm.
The agent class is included in `dqn_agent.py` while the Deep Neural Network models used by the agent are included in the `model.py`.

## ToDo list

Much more stuff can be done to around this project.

### Training Algorithm improvements

This Deep Q-Learning algorithm can be still improved with proved extensions.
Here the planned implementations:
1.Prioritized replay
2.Distributional DQN

### Deep Analysis of the Paramenters and Architecture

Wide analysis of the impact of the training parameters and model architecture on the training performance. For that, I will setup a specific notebook.

### Step to Pixel Based State Space

Use the *Navigation_Pixels.ipynb* and adapt the agent code to solve the banana collection enviroment using raw pixels. That will require mainly the modification of the `model.py` to include convolutional layers.
