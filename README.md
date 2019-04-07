# Project : Continuous control



![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

### Details

The "continuous control" project of the Udacity's Deep Reinforcement learning Nanodegree consist in training  a double-jointed arm to maintain its position at a given target location for as many time steps as possible. The arm is given a reward of +0.1 each time its hand reaches the goal location.

The observation space is comprised of 33 variables corresponding to position, rotation, velocity and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints (whose values lies in [-1,1] )

Two versions are provided here. In the first one, the learning task is performed by one single agent whereas in the second a total of 20 agents are involved.  Here my PPO implementation is solving the first version but one simple change is required for adapting it to the second one (namely adding a new dimension of size 20 to all the tensors).

The task is episodic. Each episode consist in generating a trajectory (sequence of states) within the Unity environment and getting a final score which is simply the sum of every single reward received along the way. 

In order for the environment to be solved, the episodes have to be repeated until getting an average score  of 30 (at least) over 100 consecutive episodes (and over all 20 agents in the case of second version).

​	

### Getting started

1. Clone the [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning) and follow the instructions in `README.md` to properly set up your python environment.
2. Download the Unity Environment from one of the links below, depending on your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) 
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) 
   - Windows (32-bits): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) 
   - Windows (64-bits): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Unzip the file.



### Instructions 

Open `Continuous_Control.ipynb` to train the agent.