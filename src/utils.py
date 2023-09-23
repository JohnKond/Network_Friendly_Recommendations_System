import numpy as np
import itertools

def generate_relevance_matrix_cached(K, symmetric):
    # Relevance matrix
    U = np.random.random((K,K))
    np.fill_diagonal(U, 0)
    if symmetric :
        U = (U + np.transpose(U))/2

    cached = np.random.choice(K, int(0.2*K), replace=False)
    return U, cached


# returns the reward of a state
def get_reward(state, cached):
    if state in cached:
        return 1
    return 0


# returns the cost of a state
def get_cost(state, cached):
    if state in cached:
        return 0
    return 1


# generates a random policy
def random_policy(K, N):
    # Initialize the policy as a dictionary
    policy = {}

    for s in range(K):
        remaining_items = list(range(K))
        remaining_items.remove(s)
        policy[s] = tuple(np.random.choice(remaining_items, N, replace=False))
    return policy


# return all the states except the current one
def possible_items(curr_state, K):
    return [range_val for range_val in range(K) if range_val != curr_state]


# generate all possible action of a state, i.e all tuples of recommendations(item_1, item_2)
def generate_actions(state, K, N):
    items = possible_items(state, K)
    actions =  list(itertools.combinations(items,N))
    return actions

# Function to calculate the average cost given the policy
def calculate_cost(policy, N, U, cached):
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
    new_list = []
    for i in list:
        new_list.append(i.cpu())
    return new_list


