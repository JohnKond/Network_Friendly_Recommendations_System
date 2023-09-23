import numpy as np
import time
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from utils import generate_actions,possible_items, get_reward, get_cost, calculate_cost, random_policy
import plots

class Q_Learning_Recommender:

    def __init__(self, K, N, U, cached, u_min, lr, gamma, num_episodes, q, a):
        
        # number of items/videos
        self.K = K

        # number of recommendations
        self.N = N

        # relevance threshold in [0,1]
        self.u_min = u_min

        # relevance matrix
        self.U = U

        # discount factor
        self.gamma = gamma

        # define cached items
        self.cached = cached

        # probability of ending session
        self.q = q

        # probability of choosing a recommended item, if the user chooses to continue the session
        self.a = a

        # probability of choosing one of the rest k items
        self.p_k = 1/(self.K-1)

        # learning rate
        self.lr = lr




        # Epsilon strategy
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000

        self.total_steps = 0

        # total number of episodes to run
        self.num_episodes = num_episodes

        self.all_actions = dict() # dict with all possible actions
        self.cost_per_ep = [] # cost per episode list
        self.reward_per_ep = [] # cost per episode list

        self.all_policies = []

        for state in range(self.K):
            self.all_actions[state] = generate_actions(state, self.K, self.N)


        # print('All actions : ')
        # print(self.all_actions)

        # Q table
        self.Q = np.zeros((self.K,len(self.all_actions[0])), dtype=object)

        self.main()


    def Q_Learning_Recommender(self):

        
        # Perform Q-learning
        for episode in range(1,self.num_episodes+1):
            state = np.random.randint(self.K)  # Initial state

            # print('-----------------------------')
            # print(f'State : {state}')

            # initialize variables
            steps = 0
            episode_cost = 0
            episode_reward = 0
            end_session = False

            # set exploration rate to explore more in the begining and decrease over time
            epsilon = min(1,(episode **(-1/3))*((self.K-1)*math.log(episode))**(1/3))

            # epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            # math.exp(-1. * self.total_steps / self.EPS_DECAY)

            while not end_session:
            
                steps += 1

                if np.random.uniform(0, 1) < epsilon:
                    # select random action
                    action_index = np.random.randint(len(self.all_actions[state]))  # Explore: choose a random action
                    # print(f'Select random action index : {action_index}')
                else:
                    # select arg max Q-value action
                    action_index = np.argmax(self.Q[state,:])
                    # print(f'Select arg max Q-value action index : {action_index}')

                self.total_steps += 1
                action = self.all_actions[state][action_index]

                # print(f'Action : {action}')

                relevance = np.sum(self.U[state, action] > self.u_min)

                if relevance == self.N :
                    if np.random.uniform(0, 1) < self.a:
                        next_state = np.random.choice(action)  # Transition to recommended item
                        # print(f'Transition to recommended item  : {next_state}')

                    else:
                        # next_state = np.random.choice(possible_items(state, self.K))  # Transition to any item in the catalog
                        next_state = np.random.randint(self.K)
                        # print(f'Transition to any item in the catalog  : {next_state}')

                else:
                    # next_state = np.random.choice(possible_items(state, self.K))  # Transition to any item in the catalog
                    next_state = np.random.randint(self.K)
                    # print(f'Transition to any item in the catalog  : {next_state}')

                # get cost of the action
                episode_cost += get_cost(next_state, self.cached)
                episode_reward += get_reward(next_state, self.cached)


                # Update Q-value for the current state-action pair
                self.Q[state][action_index] = self.Q[state][action_index] + self.lr * (get_reward(next_state, self.cached) + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state][action_index])

                # print('Q table :')
                # print(self.Q)


                # Check if the episode is done
                end_session = np.random.uniform(0, 1) < self.q
                state = next_state


            self.cost_per_ep.append(episode_cost / steps)
            self.reward_per_ep.append(episode_reward / steps)

    def extract_optimal_policy(self):
        best_policy = dict()
        for state in range(self.K):
            best_policy[state] = self.all_actions[state][np.argmax(self.Q[state,:])]
        return best_policy


    def plot_mean_cost(self, window_size):
        # Calculate the number of windows
        num_windows = len(self.cost_per_ep) // window_size

        # Calculate the average cost for each window
        average_costs = []
        for i in range(num_windows):
            start_index = i * window_size
            end_index = (i + 1) * window_size
            window_costs = self.cost_per_ep[start_index:end_index]
            average_cost = np.mean(window_costs)
            average_costs.append(average_cost)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(average_costs, color = 'red')
        plt.xlabel('Episode')
        plt.ylabel('Average Cost')
        plt.title('Average Cost per Episode')
        plt.show()
        
        

    def plot_cost(self):
        episodes = [*range(1,self.num_episodes+1,1)]
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(np.array(self.cost_per_ep))
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.title('Total Cost per Episode')
        plt.show()


    def plot_reward(self):
        episodes = [*range(1,self.num_episodes+1,1)]
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(np.array(self.reward_per_ep))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Total Reward per Episode')
        plt.show()



    # Function to calculate moving average
    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    
    def plot_average_reward(self, window_size):
        # Calculate moving average with a window size of 10 episodes
        moving_avg_reward = self.moving_average(self.reward_per_ep, window_size)

        # Plot the moving average
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(moving_avg_reward, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Moving Average Reward')
        plt.title('Moving Average Reward per Episode')
        plt.show()




    def log(self):
        tb = pt()
        tb.title = 'Q-Learning Recommender Policy'
        tb.field_names = ["State","Action", "Relevance"]

        for key, value in self.best_policy.items():
            tb.add_row([key,value, f'{self.U[key,value]}'])
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
        QL_start_time = time.time()
        self.Q_Learning_Recommender()
        print("Q-Learning execution time : --- %s seconds ---" % (time.time() - QL_start_time))

        self.best_policy = self.extract_optimal_policy()

        self.log()
        plots.plot_policy_evaluation_heatmap(self.best_policy, self.U, self.u_min, self.cached, self.N, 'Q-Learning')
        # self.plot_reward()
        # self.plot_average_reward(window_size=int(self.num_episodes / 100))
    

        # for key, value in best_policy.items():
            # print(f'{key} : {value} = [{self.U[key,value[0]]}  {self.U[key,value[1]]}]')