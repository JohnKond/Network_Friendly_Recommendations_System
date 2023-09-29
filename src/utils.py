import numpy as np
import itertools

def generate_relevance_matrix_cached(K, symmetric):
    """
    Generate a relevance matrix with random values and a specified structure.

    Parameters:
    - K (int): Number of items.
    - symmetric (bool): Flag indicating whether the matrix should be symmetric.

    Returns:
    - U (np.ndarray): Relevance matrix.
    - cached (np.ndarray): Cached items.    
    """
    # Relevance matrix
    U = np.random.random((K,K))
    np.fill_diagonal(U, 0)
    if symmetric :
        U = (U + np.transpose(U))/2

    cached = np.random.choice(K, int(0.2*K), replace=False)
    return U, cached


# returns the reward of a state
def get_reward(state, cached):
    """
    Get the reward of a state based on whether it is in the cached items.

    Parameters:
    - state (int): Current state.
    - cached (np.ndarray): Cached items.

    Returns:
    - int: Reward (1 if the state is cached, 0 otherwise).
    """
    if state in cached:
        return 1
    return 0


# returns the cost of a state
def get_cost(state, cached):
    """
    Get the cost of a state based on whether it is in the cached items.

    Parameters:
    - state (int): Current state.
    - cached (np.ndarray): Cached items.

    Returns:
    - int: Cost (0 if the state is cached, 1 otherwise).
    """
    if state in cached:
        return 0
    return 1


# generates a random policy
def random_policy(K, N):
    """
    Generate a random policy for the given number of items and recommendations.

    Parameters:
    - K (int): Number of items.
    - N (int): Number of recommendations.

    Returns:
    - dict: Random policy.
    """
    # Initialize the policy as a dictionary
    policy = {}

    for s in range(K):
        remaining_items = list(range(K))
        remaining_items.remove(s)
        policy[s] = tuple(np.random.choice(remaining_items, N, replace=False))
    return policy


# return all the states except the current one
def possible_items(curr_state, K):
    """
    Return all the states except the current one.

    Parameters:
    - curr_state (int): Current state.
    - K (int): Number of items.

    Returns:
    - list: List of possible items.
    """

    return [range_val for range_val in range(K) if range_val != curr_state]


# generate all possible action of a state, i.e all tuples of recommendations(item_1, item_2)
def generate_actions(state, K, N):
    """
    Generate all possible actions (tuples of recommendations) for a given state.

    Parameters:
    - state (int): Current state.
    - K (int): Number of items.
    - N (int): Number of recommendations.

    Returns:
    - list: List of possible actions.
    """
    items = possible_items(state, K)
    actions =  list(itertools.combinations(items,N))
    return actions

# Function to calculate the average cost given the policy
def calculate_cost(policy, N, U, cached):
    """
    Calculate the average cost and average reward given a policy and other parameters.

    Parameters:
    - policy (dict): Policy.
    - N (int): Number of recommendations.
    - U (np.ndarray): Relevance matrix.
    - cached (np.ndarray): Cached items.

    Returns:
    - float: Average cost.
    - float: Average reward.
    """
    total_cost = 0
    for i in range(len(U)):
        action = policy[i]
        for i in range(N):
            rec_item = action[i]

            if rec_item not in cached:
                # todo add cost for relevance
                total_cost += 1
    avg_cost = total_cost / (N*len(U))
    avg_reward = ((N*len(U)) - total_cost) / (N*len(U))
    return avg_cost, avg_reward


def copy_list_cpu(list):
    """
    Copy a list while moving elements to CPU.

    Parameters:
    - lst (list): List to copy.

    Returns:
    - list: Copied list with elements on CPU.
    """
    new_list = []
    for i in list:
        new_list.append(i.cpu())
    return new_list


