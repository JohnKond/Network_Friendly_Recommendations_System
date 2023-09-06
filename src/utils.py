import numpy as np

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
def generate_actions(state, K):
    items = possible_items(state, K)
    actions =  list(itertools.combinations(items,2))
    return actions

# Function to calculate the total cost given the policy
def calculate_cost(policy, U, cached):
    total_cost = 0
    for i in range(len(U)):
        action = policy[i]
        rec_item1 = action[0]
        rec_item2 = action[1]

        if rec_item1 not in cached:
            total_cost = total_cost + 1
        if rec_item2 not in cached:
            total_cost = total_cost + 1

    return total_cost


def copy_list_cpu(list):
    new_list = []
    for i in list:
        new_list.append(i.cpu())
    return new_list

