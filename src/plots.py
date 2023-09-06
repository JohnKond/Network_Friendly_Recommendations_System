import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_cost_reward(cost_per_episode, reward_per_episode, num_episodes):
    horizon = [*range(1, num_episodes+1, 1)]
    cum_costs =  np.cumsum(cost_per_episode) / horizon
    cum_rewards =  np.cumsum(reward_per_episode) / horizon
    
    # Normalize the cumulative cost to [0, 1]
    min_cost = min(cum_costs)
    max_cost = max(cum_costs)
    normalized_cost = (cum_costs - min_cost) / (max_cost - min_cost)

    # Normalize the cumulative reward to [0, 1]
    min_reward = min(cum_rewards)
    max_reward = max(cum_rewards)
    normalized_reward = (cum_rewards - min_reward) / (max_reward - min_reward)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(normalized_cost,color="orange")
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.title('Cumulative Cost per Episode')


    plt.subplot(1, 2, 2)
    plt.plot(normalized_reward,color="green")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Cumulative Reward per Episode')

    plt.tight_layout()
    plt.show()


def plot_multiple_cost(K_list, cost_per_episode_list, num_episodes):

    horizon = [*range(1, num_episodes+1, 1)]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))
    
    for cost_per_episode in cost_per_episode_list:
        # Calculate the cumulative cost and normalize in respect to the horizon
        cum_cost = np.cumsum(cost_per_episode) / horizon
    
        # Normalize cost to scale [0-1]
        min_cost = min(cum_cost)
        max_cost = max(cum_cost)
        normalized_cost = (cum_cost - min_cost) / (max_cost - min_cost)
        
        plt.plot(normalized_cost)
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.title('DQN - Cumulative Cost per Episode')

    plt.legend(K_list)
    plt.tight_layout()
    plt.show()


def plot_multiple_reward(reward_per_episode_list, num_episodes):
    
    horizon = [*range(1, num_episodes+1, 1)]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 5))

    for reward_per_episode in reward_per_episode_list:
        cum_reward = np.cumsum(reward_per_episode) / horizon

        # Normalize the cumulative reward to [0, 1]
        min_reward = min(cum_reward)
        max_reward = max(cum_reward)
        normalized_reward = (cum_reward - min_reward) / (max_reward - min_reward)

        plt.plot(normalized_reward,color="green")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Cumulative Reward per Episode')

    plt.tight_layout()
    plt.show()    
