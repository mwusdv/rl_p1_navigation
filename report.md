# Reinforcement Learning: Navigation Project

## Learning Algorithm
### Network Structure
The backbone neural network is a feedforward neural network with two hidden layers. 
* **Input layer**: The input to the neural network is the current state in the banana environment, which is a 37 dimensional vector.
* **Hidden layers**: There are two hidden layers. Both have 64 neurons.
* **Output layer**:  The output is the Q-score correpsonding to the input state. Therefore the output is a 4-dimensional vector, which is the same of the number of possible actions.

### Training Algorithm
There are several comoponents in the training algorithm:

1. **Replay Buffer**. This is a `deque` buffer recording the playing experiences. Each record is a **experience tuple** `(S, A, R, S', done)` consists of `state`, `action`, `reward`, `next_state` and `done` of the environment. The agent plays according to current neural network parameters. After every certain steps, a batch of experience tuples are randomly sampled from the replay buffer, which are used to update the neural network parameters. 

2. **TD Learning**. The TD learning algorithm is used to update the neural netork structure. Namely, we hope for each experience tuple `(S, A, R, S', done)`, the Q score `Q(S, A)` can be closed to the following target Q score
    
        TD-target = R + GAMMA * max_a Q(S', a) * (1-done) 

3. **Fixed Q-Targets**. Since both target and current Q sore are calculated with DQN, to make the training stable, two DQNs are used in the training process. In particular, the following upate rule is used for the experience tuple `(S, A, R, S', done)`:

        delta_w=lr * (TD-target -Q(S,A,w))*grad_w Q(S,A,w)
    where,

        TD-target = R + GAMMA * max_a Q(S', a, w_) * (1-done)

    where w_ are the weights of a separate target network that are not changed during the learning step.

## Results
1. Plot of rewards
![Plot of rewards](./scores.jpg')