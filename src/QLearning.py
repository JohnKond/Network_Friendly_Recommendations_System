import numpy as np
import time
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from utils import generate_actions
import plots

class Q_Learning_Recommender:

    def __init__(self, env, lr, gamma, num_episodes):
        """
        Initialize the Q-Learning Recommender.

        Parameters:
        - env: The environment.
        - lr: The learning rate.
        - gamma: The discount factor.
        - num_episodes: The total number of episodes.
        """

        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.num_episodes = num_episodes

        self.all_actions = dict() # dict with all possible actions
        self.cost_per_ep = [] # cost per episode list
        self.reward_per_ep = [] # cost per episode list
        self.all_policies = []

        for state in range(self.env.K):
            self.all_actions[state] = generate_actions(state, self.env.K, self.env.N)

        # Q table
        self.Q = np.zeros((self.env.K,len(self.all_actions[0])), dtype=object)

        self.main()


    def Q_Learning_Recommender(self):
        """
        Perform Q-learning for the recommender.
        """
        
        # Perform Q-learning
        for episode in range(1,self.num_episodes+1):
            state = self.env.reset()
            
            # Initialize episode flag variable
            done  = False
            
            # set exploration rate to explore more in the begining and decrease over time
            epsilon = min(1,(episode **(-1/3))*((self.env.K)*math.log(episode))**(1/3))

            while not done:

                if np.random.uniform(0, 1) < epsilon:
                    # select random action
                    action_index = np.random.randint(len(self.all_actions[state]))  # Explore: choose a random action
                else:
                    # select arg max Q-value action
                    action_index = np.argmax(self.Q[state,:])

                action = self.all_actions[state][action_index]
                next_state, reward, done = self.env.step(state, action)

                # Update Q-value for the current state-action pair
                self.Q[state][action_index] += self.lr * (reward + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state][action_index])

                # Jump to next state
                state = next_state

            
            self.cost_per_ep.append(self.env.episode_cost / self.env.steps)
            self.reward_per_ep.append(self.env.episode_reward / self.env.steps)

    def extract_optimal_policy(self):
        """
        Extract the optimal policy from the Q-table.

        Returns:
        - best_policy: The optimal policy.
        """

        best_policy = dict()
        for state in range(self.env.K):
            best_policy[state] = self.all_actions[state][np.argmax(self.Q[state,:])]
        return best_policy


    def log(self):
        """
        Log the Q-Learning Recommender policy.
        """
        tb = pt()
        tb.title = 'Q-Learning Recommender Policy'
        tb.field_names = ["State","Action", "Relevance"]

        for key, value in self.policy.items():
            tb.add_row([key,value, f'{self.env.U[key,value]}'])
        print(tb)



    def main(self):
        """
        Main function to execute Q-Learning Recommender.
        """
        QL_start_time = time.time()
        self.Q_Learning_Recommender()
        QL_end_time = time.time()
        self.policy = self.extract_optimal_policy()
        self.log()
        print("Q-Learning execution time : --- %s seconds ---" % (QL_end_time - QL_start_time))
        print('Average reward: ', np.sum(self.reward_per_ep) / self.num_episodes)
        plots.plot_policy_evaluation_heatmap(self.policy, self.env.U, self.env.u_min, self.env.cached, self.env.N, 'Q-Learning')
        self.plot_average_reward(window_size=int(self.num_episodes / 100))