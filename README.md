
![](reacher.gif)

### Introduction 

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving The Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Installation

Step 1: Clone the repo

The repo includes the Unity Environment for Linux OS

Step 2: Install Dependencies

Easiest way is to create an anaconda environment that contains all the required dependencies to run the project. Other than the modules that come with the Anaconda environment, Pytorch and unityagents are required.

```
conda create --name Reacher python=3.6
source activate Reacher
conda install -y pytorch -c pytorch
pip install unityagents
```

## Training

To train the agent that learns how to solve the environment, simply run **main.py**. This will start the training process with the default hyperparameters given to the model. When the environment is solved, the script saves the model parameters and also outputs a couple of graphs that shows the rewards per episode, and the average rewards last 100 episodes.

Weights for the model that successfully achived an average score of 30+ over 100 episodes are included. The following codes an be used to load the weights for the successful model.

```
policy = Agent(state_size, action_size, action_sigma=0.01, seed=12345)

state_dict = torch.load('actor_fin.pth')
state_dict = torch.load('critic0_fin.pth')
state_dict = torch.load('critic1_fin.pth')

policy.actor.load_state_dict(state_dict)
policy.critic0.load_state_dict((state_dict))
policy.critic1.load_state_dict((state_dict))
```
