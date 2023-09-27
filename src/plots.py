import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure
import seaborn as sns


def plot_policy_evaluation_heatmap(policy, U, u_min, cached, N, alg):
    K = len(policy)
    evaluation_values = []  # List to store evaluation values

    for state, action in policy.items():
        if np.sum(U[state, action] > u_min) == N:
            cached_ratio = sum([1 if item in cached else 0 for item in action]) / N
            if cached_ratio == 1:
                evaluation_values.append(0)  # Green
            elif cached_ratio >= 0.5:
                evaluation_values.append(1)  # Light green
            else:
                evaluation_values.append(2)  # Lighter green
        else:
            evaluation_values.append(3)  # Red

    evaluation_values = np.array(evaluation_values).reshape(1, -1)

    if len(evaluation_values) == 0:
        print("No data to display.")
        return

    # Define custom colors for the heatmap and corresponding labels
    # colors = ["#2ca25f", "#90EE90", "#98FB98", "orangered"]  # Green, Light green, Lighter green, Red
    colors = ["#2ca25f", "#90EE90", "lightblue", "orangered"]  # Green, Light green, Lighter green, Red
    labels = ["All Relevant, All Cached", "All Relevant, Some Cached", "All Relevant, Few Cached", "All Irrelevant"]
    bounds = [0, 0.9, 1.9, 2.9, 3.9]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    
    fig = plt.figure(figsize=(K+1, 1.2), dpi=100)  # Adjust the figure size
    # fig = plt.figure(figsize=(1.2, K+1), dpi=100)  # Adjust the figure size
    # Create the heatmap
    sns.heatmap(evaluation_values,linewidth=.5, cmap=cmap, norm=norm, annot=False, cbar=False, cbar_kws={"ticks": bounds, "label": labels})

    

    fig.subplots_adjust(left=0.1, right=0.9, top=0.699, bottom=0.426)


    # Set labels and title
    # plt.xlabel('State')
    # plt.title(f'Evaluation Heatmap - {alg}')

    # Remove y-axis ticks and labels
    plt.yticks([])
    plt.ylim([0,0.5])
    plt.show()
    # fig.tight_layout()
    fig.savefig(f"{alg}_heat_{K}_{N}.png",dpi=100)


    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.imshow(evaluation_values, cmap=cmap)
    # ax.set_xticks([*range(0,K)])
    # ax.set_ybound([0,0.5])
    # Show the plot
    # plt.show()
    return


def policy_heatmap(policy, cashed_items, u_min):
    """
    Creates a heatmap based on the policy dictionary, with red indicating not cashed and green indicating cashed items.

    Args:
        policy (dict): A dictionary representing the policy, where keys are recommendation indices and values are recommendation items.
        cashed_items (list or set): A list or set containing the cashed items.

    Returns:
           None. Displays the heatmap.
    """
    # Extract policy keys and values
    keys = list(policy.keys())
    values = list(policy.values())

    # Create the heatmap array
    heatmap = np.zeros((len(max(values, key=len)), len(keys)), dtype=int)

    # Fill the heatmap array with 1 for cashed items
    for j, item_list in enumerate(values):
        for i, item in enumerate(item_list):
            if item in cashed_items:
                heatmap[i, j] = 1

    # Create the heatmap plot with numbers inside the cells
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(heatmap, cmap='RdYlGn')

    # Add number annotations inside the cells
    for j in range(len(values)):
        for i in range(len(values[j])):
            ax.text(j, i, values[j][i], ha='center', va='center')

    # Add keys to the bottom of the heatmap
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_xlabel('States')

    # Remove ticks and labels from the y-axis
    ax.set_yticks([])
    ax.set_ylabel('Recommendations')

    # Set the plot title
    ax.set_title(f'Policy (Umin: {u_min})')

    # Show the plot
    plt.show()


def relevance_heatmap(relevance_matrix, threshold):
    """
    Creates a heatmap for the relevance matrix by applying a threshold.

    Args:
        relevance_matrix (numpy.ndarray): 2D array representing the relevance matrix.
        threshold (float): The threshold value for determining cyan or magenta color.

    Returns:
        None. Displays the heatmap plot.

    """
    # Apply threshold to the relevance matrix
    masked_matrix = np.where(relevance_matrix > threshold, 1, 0)

    # Create the colormap with cyan and magenta colors
    cmap = plt.cm.colors.ListedColormap(['Purple', 'Yellow'])

    # Create the heatmap plot
    plt.imshow(masked_matrix, cmap=cmap, interpolation='nearest')


    # Add title with threshold value
    plt.title(f"Relevance Matrix (Umin: {threshold})")

    # Show the plot
    plt.show()


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


def plot_multiple_reward(reward_per_episode_list, num_episodes, alg, K, N):
    """
    Plot the cumulative reward per episode for multiple algorithms.

    Args:
        reward_per_episode_list (list of lists): List of lists containing reward values per episode for each algorithm.
        num_episodes (int): Total number of episodes.
        alg (list of str): List of algorithm names (e.g., ["DQN", "Q-Learning"]).
        K (int): Total number of content items.
        N (int): Number of recommended items.
    """
    horizon = [*range(1, num_episodes+1, 1)]
    plt.style.use('seaborn-v0_8-whitegrid')
    # fig = plt.figure(figsize=(12, 5))
    cum_reward_list = []

    for reward_per_episode in reward_per_episode_list:
        # cum_reward = np.cumsum(reward_per_episode) / horizon
        plt.plot(reward_per_episode,color="green")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
    plt.title(f'{alg} - Reward per Episode')
    plt.tight_layout()
    plt.show()


# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_multiple_average_reward(reward_per_episode_list, window_size, num_episodes, legend, alg):
    
    plt.style.use('seaborn-v0_8-whitegrid')

    for reward_per_episode in reward_per_episode_list:
        
        # moving_avg_reward = np.cumsum(reward_per_episode) / [*range(1,num_episodes+1)]

        # Calculate moving average with a window size
        moving_avg_reward = moving_average(reward_per_episode, window_size)

        # Plot the moving average
        plt.plot(moving_avg_reward)

    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'{alg} Moving Average Reward per Episode')
    plt.legend(legend)
    plt.show()

