import numpy as np
import time
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from utils import generate_actions,possible_items, get_reward, get_cost, calculate_cost, random_policy
import plots

class Q_Learning_Recommender:

    def __init__(self, env, lr, gamma, num_episodes):
        

        self.env = env

        self.gamma = gamma

        # learning rate
        self.lr = lr


        # total number of episodes to run
        self.num_episodes = num_episodes

        self.all_actions = dict() # dict with all possible actions
        self.cost_per_ep = [] # cost per episode list
        self.reward_per_ep = [] # cost per episode list

        self.all_policies = []

        for state in range(self.env.K):
            self.all_actions[state] = generate_actions(state, self.env.K, self.env.N)



        # print('All actions : ')
        # print(self.all_actions)

        # Q table
        self.Q = np.zeros((self.env.K,len(self.all_actions[0])), dtype=object)

        self.main()


    def Q_Learning_Recommender(self):

        
        # Perform Q-learning
        for episode in range(1,self.num_episodes+1):
            # state = np.random.randint(self.env.K)  # Initial state
            state = self.env.reset()

            # print('-----------------------------')
            # print(f'State : {state}')
            
            # Initialize episode flag variable
            done  = False
            

            # set exploration rate to explore more in the begining and decrease over time
            epsilon = min(1,(episode **(-1/3))*((self.env.K)*math.log(episode))**(1/3))
    

            while not done:

                # print(f'State : {state}')
                if np.random.uniform(0, 1) < epsilon:
                    # select random action
                    action_index = np.random.randint(len(self.all_actions[state]))  # Explore: choose a random action
                    # print(f'Select random action index : {action_index}')
                else:
                    # select arg max Q-value action
                    action_index = np.argmax(self.Q[state,:])
                    # print(f'Select arg max Q-value action index : {action_index}')

                action = self.all_actions[state][action_index]
                # print('Action : ',action)

                next_state, reward, done = self.env.step(state, action)


                # Update Q-value for the current state-action pair
                self.Q[state][action_index] += self.lr * (reward + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state][action_index])

                # Jump to next state
                state = next_state

            
            self.cost_per_ep.append(self.env.episode_cost / self.env.steps)
            self.reward_per_ep.append(self.env.episode_reward / self.env.steps)

    def extract_optimal_policy(self):
        best_policy = dict()
        for state in range(self.env.K):
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

        for key, value in self.policy.items():
            tb.add_row([key,value, f'{self.env.U[key,value]}'])
        print(tb)



    def main(self):
        QL_start_time = time.time()
        self.Q_Learning_Recommender()
        print("Q-Learning execution time : --- %s seconds ---" % (time.time() - QL_start_time))

        self.policy = self.extract_optimal_policy()
        self.log()
        plots.plot_policy_evaluation_heatmap(self.policy, self.env.U, self.env.u_min, self.env.cached, self.env.N, 'Q-Learning')
        # self.plot_average_reward(window_size=int(self.num_episodes / 100))