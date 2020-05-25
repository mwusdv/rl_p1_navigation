# Reinforcement Learning: Navigation Project

## Project Details
For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic. **The envionment is considered as solved if the agent can get an average score of +13 over 100 consecutive episodes.**


## Getting Started
In the project, we need the Unity environment. We can download it that has already been built.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the rl_p1_navigation/ folder and unzip (or decompress) the file.

## Instructions
There are three python files in the project repo:
* `common.py`:   contains code for initializing and playing in the banana environement.

* `train.py`: contains the code for training the DQN agent. To launch the code, use 

            python train.py dqn_fname
    
    The last parameter `dqn_fname` is the name of the file where the trained agent to be stored.

* `test.py`: contains the code to play the banana environment, either randomly or with a trained DQN agent. To launch the code:

            python test.py dqn_fname

    The last parameter `dqn_fname` is optional. If it is given, the code will play the trained agent stored in that file, otherwise the code will play the environment randomly.