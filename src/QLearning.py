import numpy as np
import random
import itertools
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt

from utils import generate_actions,possible_items, get_reward, calculate_cost, random_policy

class Q_Learning_Recommender:

    def __init__(self, K, N, u_min, lr, gamma, num_episodes, q, a, symmetric):
        # random.seed(seed_no)
        # np.random.seed(seed_no)

        # number of items/videos
        self.K = K

        # number of recommendations
        self.N = N

        # relevance threshold in [0,1]
        self.u_min = u_min

        # discount factor
        self.gamma = gamma

        # number of cached items
        self.C = int(0.2 * self.K)

        # define cached items
        self.cached = np.random.choice(K, self.C, replace=False)

        # probability of ending session
        self.q = q

        # probability of choosing a recommended item, if the user chooses to continue the session
        self.a = a

        # probability of choosing one of the rest k items
        self.p_k = 1/(self.K-1)

        # Relevance matrix
        self.U = np.random.random((self.K, self.K))
        np.fill_diagonal(self.U, 0)

        if symmetric :
            self.U = (self.U + np.transpose(self.U))/2

        # learning rate
        self.lr = lr

        # total number of episodes to run
        self.num_episodes = num_episodes

        self.all_actions = dict() # dict with all possible actions
        self.cost_per_ep = [] # cost per episode list

        self.all_policies = []

        for state in range(self.K):
            self.all_actions[state] = generate_actions(state, self.K)


        # Q table
        self.Q = np.zeros((self.K,len(self.all_actions[0])), dtype=object)

        self.main()


    def Q_Learning_Recommender(self):
        states_visited = set()
        # Perform Q-learning
        for episode in range(1,self.num_episodes+1):
            state = np.random.randint(self.K)  # Initial state

            end_session = False
            curr_cost = 0
            # set exploration rate to explore more in the begining and decrease over time
            epsilon = min(1,(episode **(-1/3))*(self.K*math.log(episode))**(1/3))
            while not end_session:
                reward = 0
                if np.random.uniform(0, 1) < epsilon:
                    action_index = np.random.randint(len(self.all_actions[state]))  # Explore: choose a random action
                    # print(f'selected for state {state} random action index: {action_index}')
                else:
                    action_index = np.argmax(self.Q[state])
                    # print(f'selected for state {state} max action index: {action_index}')


                action = self.all_actions[state][action_index]
                relevance = np.sum(self.U[state, action] > self.u_min)

                if relevance == self.N :
                    if np.random.uniform(0, 1) < self.a:
                        next_state = np.random.choice(action)  # Transition to recommended item

                    else:
                        next_state = np.random.choice(possible_items(state, self.K))  # Transition to any item in the catalog

                else:
                    next_state = np.random.choice(possible_items(state, self.K))  # Transition to any item in the catalog



                # Update Q-value for the current state-action pair
                self.Q[state][action_index] = (1-self.lr)*self.Q[state][action_index] + self.lr * (get_reward(next_state, self.cached) + self.gamma * np.max(self.Q[next_state]))
                # self.Q[state][action_index] = (1-self.lr)*self.Q[state][action_index] + self.lr * (reward + self.gamma * np.max(self.Q[next_state]))



                # Check if the episode is done
                end_session = np.random.uniform(0, 1) < self.q



            # extract new policy from the estimated Q-table
            new_policy = self.extract_optimal_policy()
            self.all_policies.append(new_policy)

            # calculate cost of the given policy normalized in respect to the number of episodes
            self.cost_per_ep.append(calculate_cost(new_policy, self.U, self.cached) / self.K) #/ self.num_episodes)

    def extract_optimal_policy(self):
        best_policy = dict()
        for state in range(self.K):
            best_policy[state] = self.all_actions[state][np.argmax(self.Q[state])]
        return best_policy


    def plot(self):
        episodes = [*range(self.num_episodes)]
        plt.plot(self.cost_per_ep)
        plt.xlabel('Episode')
        plt.ylabel('Total Cost')
        plt.title('Total Cost per Episode')
        plt.show()


    def log(self):
        tb = pt()
        tb.title = f'Q-Learning Recommender Policy - cost = {self.cost_per_ep[-1]}'
        tb.field_names = ["State","Action", "Relevance"]

        for key, value in self.best_policy.items():
            tb.add_row([key,value, f'{self.U[key,value[0]]}  {self.U[key,value[1]]}'])
        print(tb)


    def plot_grid(self):
        colors = np.zeros(( self.K,len(self.all_policies)))

        i = 0
        for policy in self.all_policies:
            for state in range(self.K):
                recommendations = policy[state]
                num_cached = sum([1 if item in self.cached else 0 for item in recommendations])
                if num_cached == 2:
                    colors[state, i] = 0  # Green
                elif num_cached == 1:
                    colors[state, i] = 0.5  # Orange
                else:
                    colors[state, i] = 1  # Red
            i += 1

        fig, ax = plt.subplots(figsize=(20, 15))
        data = ax.imshow(colors, cmap="Paired_r", origin="lower", vmin=0)
        # plt.title('Cached Recommendations in Q-Learning')
        plt.xlabel('Number of Episodes')
        plt.ylabel('States')
        ax.set_xticks(np.arange(i+1)-0.5, minor=True)
        ax.set_yticks(np.arange(self.K+1)-0.5, minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", size=0)
        # plt.legend(['Cached','Uncached'])
        # plt.colorbar(data)
        plt.show()


    def main(self):
        # print('Most relevant items for each state :')
        # for state in range(len(self.U)):
            # most_rel = np.argsort(self.U[state])
            # two_most_rel = most_rel[-2:]
            # print(f'{state} : {two_most_rel} = [{self.U[state,two_most_rel[0]]}  {self.U[state,two_most_rel[1]]}]')

        print(f'\n Cached items : {sorted(self.cached)}\n')
        self.Q_Learning_Recommender()
        self.best_policy = self.extract_optimal_policy()

        # for key, value in best_policy.items():
            # print(f'{key} : {value} = [{self.U[key,value[0]]}  {self.U[key,value[1]]}]')