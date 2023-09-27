import numpy as np
import random
import math
import utils
import plots

class RecommendationEnvironment():
    def __init__(self, K, N, u_min, q, alpha):
        """
        Initializes the RecommendationEnvironment.

        Parameters:
        - K (int): Total number of content items.
        - N (int): Number of recommended items.
        - U (numpy.ndarray): A matrix representing the relevance scores between content items.
        - cached (list): A list of cached items.
        - u_min (float): The relevance threshold.
        - q (float): Probability of ending the session.
        - alpha (float): Probability of choosing a recommended item.
        - C (int): The number of cached items.
        - symmetric (bool): Boolean indicating whether the relevance matrix is symmetric.
        """

        self.K = K  
        self.N = N  
        self.u_min = u_min
        self.q = q 
        self.alpha = alpha
        self.U, self.cached = utils.generate_relevance_matrix_cached(K,True)
        print('Cached items: ', sorted(self.cached))
        self.steps = 0
        self.episode_cost = 0
        self.episode_reward = 0

        # plots.relevance_heatmap(self.U,u_min)  

    def reset(self):
        """
        Resets the environment to a new episode and returns the initial state.

        Returns:
        - int: The initial state.
        """
        state = np.random.choice(self.K)
        self.steps = 0
        self.episode_cost = 0
        self.episode_reward = 0
        return state

    def step(self, state, action):
        """
        Takes a step in the environment based on the given action.

        Parameters:
        - action (list): A list of recommended items.

        Returns:
        - int: The new state.
        - float: The reward.
        - bool: True if the episode is done, False otherwise.
        """
        done = False


        # Calculate relevance of recommended items
        relevance = np.sum(self.U[state, action] > self.u_min)


        # Check if all recommended items are relevant
        if relevance == self.N :
            # Check if the user chooses a recommended item
            if np.random.uniform(0, 1) < self.alpha:
                next_state = np.random.choice(action)
                # print('Choose recommendation : ',next_state)
            else:
                # User chooses any item from the entire catalog with uniform probability
                next_state = np.random.randint(self.K)
                # print('Choose other item : ',next_state)
        else:
            # User chooses any item from the entire catalog with uniform probability
            next_state = np.random.randint(self.K)
            # print('Choose other item : ',next_state)

        # Calculate cost based on whether the selected item is cached or not
        cost = 0 if next_state in self.cached else 1
        reward = 1 if next_state in self.cached else 0

        self.episode_cost += cost
        self.episode_reward += reward
        self.steps += 1

        # Check if the session ends with probability q
        if np.random.rand() < self.q:
            done = True

        return next_state, reward, done