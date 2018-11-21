
import torch
from util import raw_score_plotter, plotter
from agent import Agent
import numpy as np
from collections import deque

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='reacher/Reacher.x86_64')
env_name = 'Reacher'
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# size of each action
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

policy = Agent(state_size, action_size, action_sigma=0.01, seed=12345)

num_episodes  = 1500000

rewards     = []
batch_size  = 128
threshold = 30.0
scores = []
scores_deque = deque(maxlen=100)
PRINT_EVERY = 5

avg_last_100 = []

for eps in range(num_episodes):

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state (for each agent)
    episode_reward = 0
    done = 0
    step = 0
    while done == 0:

        if eps % 100 == 0:
            '''For every 100 episodes we perform the task without any noise
            The authors talk about doing this during training to gauge how well
            the agent has learned to solve the environment. Although running 1 episode might not be enough'''
            action = policy.act(state, False)
            action = np.clip(action, -1, 1)
        else:
            action = policy.act(state)
            action = np.clip(action, -1, 1)

        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        episode_reward += reward

        step += 1

        if eps % 100 != 0:
            '''
            Chose not to use the data from the episode without any noise. 
            
            '''
            policy.add_to_memory(state, action, reward, next_state, done)
            if policy.memory.__len__() > batch_size:
                policy.update(step) #passes in steps so the actor knows when to update.

        state = next_state

        if done:
            break

    scores_deque.append(episode_reward)
    avg_last_100.append(np.mean(scores_deque))
    scores.append(episode_reward)

    if eps % 100 == 0:
        print("\nThis Episode :", eps)
        print("\nRewards without noise is: ", episode_reward)
        torch.save(policy.actor.state_dict(), 'actor.pth')
        torch.save(policy.critic0.state_dict(), 'critic0.pth')
        torch.save(policy.critic1.state_dict(), 'critic1.pth')
        torch.save(policy.target_actor.state_dict(), 'target_actor.pth')
        torch.save(policy.target_critic0.state_dict(), 'target_critic0.pth')
        torch.save(policy.target_critic1.state_dict(), 'target_critic1.pth')

    if eps % PRINT_EVERY == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(eps, avg_last_100[-1]))

    if avg_last_100[-1] >= threshold:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(eps - 100, avg_last_100[-1]))
        torch.save(policy.actor, 'actor_fin.pth')
        torch.save(policy.critic0, 'critic0_fin.pth')
        torch.save(policy.critic1, 'critic1_fin.pth')
        torch.save(policy.target_actor, 'target_actor_fin.pth')
        torch.save(policy.target_critic0, 'target_critic0_fin.pth')
        torch.save(policy.target_critic1, 'target_critic1_fin.pth')
        break

raw_score_plotter(scores)
plotter(env_name, len(scores), avg_last_100, threshold)