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

Beside the Unity enviroment, Python 3.6 must be available with the Unity ML-Agents [(see this link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) installed, and few more packages such as 'Numpy', 'Pytorch', 'collections', 'pandas' and 'matplotlib'.
 
## Instructions

Instruction to run the code are provided along with the "Navigation.ipynb".
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
10. An example of a trained agent in the bananas colletion enviroment.

## ToDo list

This Deep Q-Learning algorithm can be still improved with proved extensions.
Here the planned implementations:
1.Prioritized replay
2.Distributional DQN

Use the *Navigation_Pixels.ipynb* and adapt the agent code to solve the banana collection enviroment using raw pixels. That will require mainly the modification of the *model.py* to include convolutional layers.
