import numpy as np
import random
import math
from collections import namedtuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from prettytable import PrettyTable as pt
import plots


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initializes the DQN (Deep Q-Network) neural network.

        Parameters:
        - state_dim (int): Dimensionality of the state input.
        - action_dim (int): Dimensionality of the action output.
        - hidden_dim (int): Dimensionality of the hidden layers.
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(1 + state_dim, 64)
        self.layer2 = nn.Linear(64, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, 1 + state_dim).

        Returns:
        - Tensor: Output tensor of shape (batch_size, action_dim).
        """
        # x = F.sigmoid(self.layer1(x))
        x = F.relu(self.layer1(x))

        # x = F.sigmoid(self.layer2(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        """
        Initialize a replay memory buffer with a given capacity.

        Args:
            capacity (int): The maximum number of transitions that the buffer can hold.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Add a new transition to the replay memory.

        Args:
            *args: A Transition tuple (state, action, next_state, reward).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the replay memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of sampled Transition tuples.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current number of transitions stored in the replay memory.

        Returns:
            int: The current size of the memory buffer.
        """
        return len(self.memory)
    



def select_action(env, state_tensor, input_tensor, epsilon, device, policy_net, total_steps):
    """
    Select an action based on an epsilon-greedy policy.

    Args:
        state_tensor (torch.Tensor): The tensor representing the current state.
        input_tensor (torch.Tensor): The tensor containing input data, including the current state.
        epsilon (float): The exploration probability (probability of selecting a random action).
        device (torch.device): The device for computation (e.g., CPU or GPU).
        policy_net (nn.Module): The neural network representing the policy.
        total_steps (int): The total number of steps taken in the environment.

    Returns:
        torch.Tensor: A tensor containing the selected recommendations.
    """

    # Generate a random number between 0 and 1
    if np.random.rand() < epsilon:
        # Select a random action with probability epsilon
        action = np.random.choice(env.K, env.N, replace=False)
        recommendations = torch.tensor(action, device=device)
    else:
        # Select the action with the highest Q-value from the policy network
        with torch.no_grad():
            q_values = policy_net.forward(input_tensor)

        q_values[state_tensor] = -np.inf
        _, recommendations = q_values.topk(env.N)

    return recommendations


# Define the Q-learning function
def q_learning(env, policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    """
    Optimization using Q-learning algorithm in DQN.

    Args:
        env: The environment.
        policy_net (nn.Module): The policy network.
        target_net (nn.Module): The target network.
        optimizer: The optimizer for updating the policy network's weights.
        memory: Replay memory for experience replay.
        batch_size (int): The size of the batch to sample from the memory.
        gamma (float): The discount factor for future rewards.
        device: The device for computation (e.g., CPU or GPU).

    Returns:
        None
    """
    # Sample a batch of experiences from the memory buffer
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.stack(batch.action,dim=0).to(device)
    reward_batch = torch.cat(batch.reward).to(device)


    rel_scores_list = [env.U[state] for state in state_batch]
    
    # Convert the list of inputs to a NumPy array
    rel_scores_array = np.array(rel_scores_list, dtype=np.float32)


    # Convert the relevance scores to a tensor
    relevance_scores_tensor = torch.tensor(rel_scores_array, dtype=torch.float32, device=device)

    # Create the input_batch by concatenating the states and relevance scores
    input_batch = torch.cat((state_batch.view(-1, 1), relevance_scores_tensor), dim=1).squeeze(1).to(device)

    # Compute the Q-values for the current state-action pairs from the policy network
    state_action_values = torch.mean(policy_net(input_batch).gather(1, action_batch),1)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    next_states_relevance_scores = np.array([env.U[state] for state in non_final_next_states])
    next_states_relevance_scores_tensor = torch.tensor(next_states_relevance_scores, dtype=torch.float32, device=device)

    # Compute the Q-values for the next state-action pairs from the target network
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_input_batch = torch.cat((non_final_next_states.view(-1, 1), next_states_relevance_scores_tensor), dim=1).squeeze(1).to(device)
    next_state_values[non_final_mask] = torch.mean(torch.topk(target_net.forward(next_state_input_batch),  env.N, dim=1)[0],1)

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
def dqn_train(env, state_dim, hidden_dim, batch_size, gamma, target_update, num_episodes, TAU, device):
    """
    Main training function for Deep Q-Network (DQN).

    Args:
        env: The environment.
        state_dim (int): The state dimension.
        hidden_dim (int): The dimension of the hidden layers in the DQN.
        batch_size (int): The size of the batch for experience replay.
        gamma (float): The discount factor for future rewards.
        target_update (int): The frequency at which to update the target network.
        num_episodes (int): The number of training episodes.
        TAU (float): Parameter for soft target network updates.
        device: The device for computation (e.g., CPU or GPU).

    Returns:
        policy_net: The trained policy network.
        rewards_per_episode (list): List of total rewards per episode during training.
        cost_per_episode (list): List of total costs per episode during training.
    """

    # define action space, which is equal to state space (K)
    action_dim = state_dim

    # Initialize the policy network and target network
    policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
    target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())


    # Define the optimizer (SparseAdam + AdamW) and experience replay memory
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
    memory = ReplayMemory(10000)

    # Initialize other variables
    rewards_per_episode = []
    cost_per_episode = []    
    total_steps = 0

    # Start the training loop
    for episode in range(1,num_episodes+1):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)

        done = False
        total_reward = 0
        steps = 0
        

        # exploration - exploitation rate, decreases over time
        epsilon = min(1,(episode **(-1/3))*(state_dim*math.log(episode))**(1/3))

        while not done:

            steps += 1

            input = np.insert(env.U[state],0,state)
            input_tensor = torch.tensor(input, dtype=torch.float, device=device)
            
            # Select an action using epsilon-greedy policy
            action = select_action(env, state_tensor, input_tensor, epsilon, device, policy_net, total_steps)

            # Take a step in the environment and observe reward and next state
            next_state, reward, done = env.step(state, action.cpu())
            reward = torch.tensor(reward, device=device).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.long, device=device).unsqueeze(0)

            # Store experience in memory
            memory.push(state_tensor, action, next_state_tensor, reward)

            # Update the total reward and cost
            total_reward += reward.item()

            # Perform Q-learning update if enough experiences are available
            if len(memory) >= batch_size:
                q_learning(env, policy_net, target_net, optimizer, memory, batch_size, gamma, device)

            # Move to the next state
            state_tensor = next_state_tensor
            state = next_state

        # Previous, just load state dict
        # target_net.load_state_dict(policy_net.state_dict())

        # Soft update neural net weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # Store the total reward for the current episode

        rewards_per_episode.append(total_reward / steps)
        total_steps += steps

    return policy_net, rewards_per_episode, cost_per_episode


def extract_policy(env, policy_net, device):
    """
    Extracts the policy (recommendations) from the trained policy network.

    Args:
        env: The environment.
        policy_net: The trained policy network.

    Returns:
        policy (dict): A dictionary mapping states to recommended items.
    """
    policy = {}
    state_tensors = torch.arange(env.K, dtype=torch.long, device=device).unsqueeze(0)  # Create a tensor for all states

    inputs = []  # Initialize a list to store input tensors

    for state in range(env.K):
        input = np.insert(env.U[state], 0, state)
        inputs.append(input)

    
    # Convert the list of inputs to a NumPy array
    input_array = np.array(inputs, dtype=np.float32)
    
    # Convert the numpy array to a tensor
    input_tensors = torch.tensor(input_array, dtype=torch.float, device=device)  # Convert the list of inputs to a tensor

    

    with torch.no_grad():
        q_values = policy_net(input_tensors)  # Compute Q-values

    q_values[state_tensors, state_tensors] = -np.inf  # Set Q-values for the current state to -inf
    _, recommendations = q_values.topk(env.N)  # Get the top-N recommendations
    recommendations_np = recommendations.cpu().numpy()  # Convert recommendations tensor to a NumPy array

    for state, recs in enumerate(recommendations_np):
        policy[state] = recs

    return policy



def print_policy(env, policy):
    """
    Print the optimal policy extracted from the DQN model.

    Args:
        env: The environment.
        policy (dict): A dictionary mapping states to recommended items.
    """
    tb = pt()
    tb.title = f'DQN Optimal Policy'
    tb.field_names = ["Watching","Recommendations","Relevance"]

    for key, value in policy.items():
        tb.add_row([key,sorted(value.tolist()), f'{env.U[key,sorted(value.tolist())]}'])
    print(tb)



def DQN_run(env, gamma, num_episodes, device) :
    """
    Run the Deep Q-Network (DQN) algorithm in a recommendation system environment.

    Parameters:
    - env: The environment.
    - gamma (float): The discount factor for future rewards.
    - num_episodes (int): Total number of training episodes.

    Returns:
    - policy (dict): Optimal policy extracted from the DQN algorithm.
    - cost_per_episode (list): Cost per episode during training.
    - reward_per_episode (list): Reward per episode during training.
    """
    
    
    # Initialize neural network variables
    batch_size = 64  # Batch size for training the DQN
    TAU = 0.005 # soft update rate
    dqn_start_time = time.time()

    # Start training
    policy_net, reward_per_episode, cost_per_episode = dqn_train(
        env = env,
        state_dim = env.K,
        hidden_dim = 128,
        batch_size = batch_size,
        gamma = gamma,
        target_update = 20,
        num_episodes=num_episodes,
        TAU = TAU,
        device = device
    )

    dqn_end_time = time.time()
    

    policy = extract_policy(env, policy_net, device)
    print_policy(env, policy)

    print("DQN : --- %s seconds ---" % (dqn_end_time - dqn_start_time))
    print('Average reward gained: ', np.sum(reward_per_episode) / num_episodes)

    if env.K <= 500 :
        plots.plot_policy_evaluation_heatmap(policy, env.U, env.u_min, env.cached, env.N, 'DQN')
    plots.plot_multiple_average_reward([reward_per_episode], int(num_episodes/10), num_episodes, ['DQN'], 'DQN')
    return policy, cost_per_episode, reward_per_episode