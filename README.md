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
Firt download the necessary zip file according to the above section. Then please modify the 16-th line in the `common.py` accordingly:

        env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86")
with the correct file name. Currently it is set as `./Banana_Linux/Banana.x86` since I worked on this project in a ubuntu system. 

To launch the training code:

            python train.py dqn_fname
    
The last parameter `dqn_fname` is the name of the file where the trained agent will be stored.

To launch the test code:

            python test.py dqn_fname

The last parameter `dqn_fname` is optional. If it is given, the code will play with the trained agent stored in that file, otherwise the code will play the environment randomly.