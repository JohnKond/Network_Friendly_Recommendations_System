import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import namedtuple, deque
from itertools import count
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import gym
from gym import spaces
from prettytable import PrettyTable as pt
from utils import generate_relevance_matrix_cached
from plots import plot_multiple_cost, plot_multiple_reward, plot_cost_reward


class RecommendationEnvironment(gym.Env):
    def __init__(self, K, N, U, cached, u_min, q, alpha, C, symmetric):
        self.K = K  # Total number of content items
        self.cached_num = int(0.2 * self.K)  # Number of cached items
        self.N = N  # Number of recommended items
        self.u_min = u_min  # Relevance threshold
        self.q = q  # Probability of ending the session
        self.alpha = alpha  # Probability of choosing a recommended item

        self.cached_items = np.random.choice(K, self.cached_num, replace=False)

        # Relevance matrix
        self.U = np.random.random((self.K, self.K))
        np.fill_diagonal(self.U, 0)

        if symmetric :
            self.U = (self.U + np.transpose(self.U))/2


        # try
        self.U = U
        self.cached_items = cached
        
        # self.relevance_matrix = np.random.rand(K, K)
        # np.fill_diagonal(self.relevance_matrix, 0)

        self.state_space = spaces.Discrete(K)
        self.action_space = spaces.Tuple([spaces.Discrete(K) for _ in range(N)])

        self.current_state = None
        self.current_step = None
        self.total_cost = 0
        self.total_reward = 0

    def reset(self):
        self.current_state = np.random.choice(self.K)
        self.current_step = 0
        self.total_cost = 0
        return self.current_state

    def step(self, action):

        done = False

        # Check if the session ends with probability q
        if np.random.rand() < self.q:
            done = True
            return self.current_state, 0, done


        # Calculate relevance of recommended items
        relevance = np.sum(self.U[self.current_state, action] > self.u_min)

        # relevances = [self.relevance_matrix[self.current_state, a] for a in action]

        # Check if all recommended items are relevant
        # if all(relevance > self.u_min for relevance in relevances):
        if relevance == self.N :
            # Check if the user chooses a recommended item
            if np.random.rand() < self.alpha:
                selected_item = np.random.choice(action)
            else:
                # User chooses any item from the entire catalog with uniform probability
                selected_item = np.random.randint(self.K)
        else:
            # User chooses any item from the entire catalog with uniform probability
            selected_item = np.random.randint(self.K)

        # Calculate cost based on whether the selected item is cached or not
        cost = 0 if selected_item in self.cached_items else 1
        reward = 1 if selected_item in self.cached_items else 0


        # just a try
        # cost = -1 if selected_item in self.cached_items else 1
        # reward = 1 if selected_item in self.cached_items else -1

        # this works fine
        # cost = 0 if selected_item in self.cached_items else 1
        # reward = 1 if selected_item in self.cached_items else -1

        self.total_cost += cost
        self.total_reward += reward

        self.current_state = selected_item
        self.current_step += 1

        return self.current_state, reward, done

    def render(self):
        pass  # You can implement a rendering function to visualize the environment if needed

    def get_total_cost(self):
        return self.total_cost



# Define the Deep Q-Network (DQN) class
class DQN(nn.Module):

    # Define the neural network layers
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Embedding(state_dim, hidden_dim, sparse = True)#nn.Linear(n_observations, 128)#
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    # Perform forward pass through the neural network
    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))
        return self.layer3(x)
    

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad(set_to_none=True)

    def step(self):
        for op in self.optimizers:
            op.step()



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        # Initialize the memory buffer with a given capacity
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Push a new experience into the memory buffer
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the memory buffer
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Return the current size of the memory buffer
        return len(self.memory)
    



def select_action(state, K, N, epsilon, device, policy_net):

    # Generate a random number between 0 and 1
    if np.random.rand() < epsilon:
        # Select a random action with probability epsilon
        action = np.random.choice(K, N, replace=False)
        recommendations = torch.tensor(action, device=device)
    else:
        # Select the action with the highest Q-value from the policy network
        with torch.no_grad():
            q_values = policy_net.forward(state)


        q_values[0][state] = -np.inf
        _, recommendations = q_values[0].topk(N)

    return recommendations


# Define the Q-learning function
def q_learning(N, policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    # Sample a batch of experiences from the memory buffer
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.stack(batch.action,dim=0).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Compute the Q-values for the current state-action pairs from the policy network
    state_action_values = torch.mean(policy_net(state_batch).gather(1, action_batch),1)    


    # Compute the Q-values for the next state-action pairs from the target network
    next_state_values = torch.zeros(batch_size, device=device)
    
    next_state_values[non_final_mask] = torch.mean(torch.topk(target_net.forward(non_final_next_states),  N, dim=1)[0],1)

    # Compute the expected Q-values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    expected_state_action_values = expected_state_action_values.detach()


    # Compute the loss using Huber loss (smooth L1 loss)
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    
    # Optimize the Q-network's weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Define the main DQN training function
def dqn_train(env, state_dim, hidden_dim, N, batch_size, gamma, num_episodes, TAU, device):
    
    # define action space, which is equal to state space (K)
    action_dim = state_dim

    # Initialize the policy network and target network
    policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
    target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    

    # Define the optimizer (SparseAdam + AdamW) and experience replay memory
    optimizer = MultipleOptimizer(optim.SparseAdam(list(policy_net.parameters())[:1], lr=1e-4), optim.AdamW(list(policy_net.parameters())[1:], lr=1e-4))
    memory = ReplayMemory(10000)

    # Initialize other variables
    rewards_per_episode = []
    cost_per_episode = []

    # Start the training loop
    for episode in range(1,num_episodes+1):
        
        state = env.reset()
        state = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        total_cost = 0

        # exploration - exploitation rate, decreases over time
        epsilon = min(1,(episode **(-1/3))*(state_dim*math.log(episode))**(1/3))

        while not done:

            # Select an action using epsilon-greedy policy
            action = select_action(state, state_dim, N, epsilon, device, policy_net)
            

            # Take a step in the environment and observe reward and next state
            next_state, reward, done = env.step(action.cpu())
            reward = torch.tensor(reward, device=device).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.long, device=device).unsqueeze(0)
            
            # Store experience in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Update the total reward and cost
            total_reward += reward.item()

            # Perform Q-learning update if enough experiences are available
            if len(memory) >= batch_size:
                q_learning(N, policy_net, target_net, optimizer, memory, batch_size, gamma, device)

        # Previous, just load state dict
        # target_net.load_state_dict(policy_net.state_dict())

        # Soft update neural net weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # Store the total reward and cost for the episode

        # print('Episode cost : ',((N*K) - total_reward) / (N*K))
        # print('Episode reward : ',((total_reward) / (N*K)))

        # Compute average reward and cost (in respect to the total number of recommendations) at the end of each episode
        rewards_per_episode.append(total_reward / (N*state_dim))
        cost_per_episode.append(((N*state_dim) - total_reward) / (N*state_dim))


    return policy_net, rewards_per_episode, cost_per_episode


def extract_policy(K, N, policy_net, device):
    policy = {}
    for state in range(K):
        state_tensor = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
        policy[state] = select_action(state_tensor, K, N, -np.inf, device, policy_net)

    tb = pt()
    tb.title = f'DQN Optimal Policy'
    tb.field_names = ["State","Action"]

    for key, value in policy.items():
        tb.add_row([key,value.tolist()]) #, f'{recommendation_environent.relevance_matrix[key,value[0]]}  {recommendation_environent.relevance_matrix[key,value[1]]}'])
    print(tb)

    return policy


def DQN_run(K, N, U, cached, u_min, alpha, q, gamma, num_episodes, device) :

    # Define hyperparameters
    batch_size = 64  # Batch size for training the DQN
    TAU = 0.005 # soft update rate

    # Initialize Recommendation System enviroment
    recommendation_environment = RecommendationEnvironment(
        K = K,
        N = N,
        U = U,
        cached=cached,
        u_min = u_min,
        q = q,
        alpha = alpha,
        C = 0.2,
        symmetric=False
    )

    print('K = ',K)
    print('N = ',N)
    print('Cached items : ',sorted(recommendation_environment.cached_items))

    # Start training
    policy_net, reward_per_episode, cost_per_episode = dqn_train(
        env = recommendation_environment,
        state_dim = K,
        hidden_dim = 128,
        N = N,
        batch_size = batch_size,
        gamma = gamma,
        num_episodes=num_episodes,
        TAU = TAU,
        device = device
    )

    policy = extract_policy(K, N, policy_net, device)
    plot_cost_reward(cost_per_episode, reward_per_episode, num_episodes)
    return cost_per_episode, reward_per_episode