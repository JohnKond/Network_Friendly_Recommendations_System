# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
from prettytable import PrettyTable as pt
from utils import generate_actions, possible_items, get_reward, calculate_cost, random_policy, generate_relevance_matrix_cached
import plots


class Policy_Iteration_Recommender:
    def __init__(self, K, N, U, cached, u_min, gamma, theta, a, q):
        
        # number of items/videos
        self.K = K

        # number of recommendations
        self.N = N

        # relevance threshold in [0,1]
        self.u_min = u_min

        # discount factor
        self.gamma = gamma

        # terminating number ~0
        self.theta = theta

        # Relevance matrix
        self.U = U

        
        # define cached items
        self.cached = cached

        # probability of ending session
        self.q = q

        # probability of choosing a recommended item, if the user chooses to continue the session
        self.a = a

        # probability of choosing one of the rest k items
        self.p_k = 1/(self.K-1)

        # total costs of the algorithm per iteration
        self.total_costs = []

        # initialize policy
        self.policy = dict()

        self.all_policies = []

        self.iterations = 0

        # start procedure
        self.main()



    def PolicyEvaluation(self, policy, max_iterations, theta):

        # Initialize the value state function
        V = np.zeros(self.K)

        for i in range(max_iterations):

            V_prev = np.copy(V)

            # iterate over all possible states / items
            for state in range(self.K):

                recommendations = policy[state]
                
                relevance = np.sum(self.U[state, recommendations] > self.u_min)

                # generate all posible next states
                possible_states = possible_items(state, self.K)

                if relevance == self.N:

                    # expected return if user selects one the recommended items
                    # user_rec = (1/self.N) * ((get_reward(rec_item1, self.cached) + self.gamma * V[rec_item1]) + (get_reward(rec_item2, self.cached) + self.gamma * V[rec_item2]))

                    user_rec = (1/self.N) * sum((get_reward(next_state, self.cached) + self.gamma * V[next_state]) for next_state in recommendations)

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    V[state] = (1-self.q) * (self.a*user_rec + (1-self.a)*user_not_rec)

                else :

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    V[state] = (1-self.q) * user_not_rec



            # convergence check
            if np.max(np.abs(V - V_prev)) <= 0.001:
                # print(f'Converged on {i} iterations')
                break

            # max_iterations check
            if i == max_iterations :
                print('Policy Evaluation : Max iterations reached')
                break
        return V



    def PolicyImprovement(self, V):
        #initialize new policy dict
        new_policy = {}

        # iterate over all possible states / items
        for state in range(self.K):

            # for max comparison
            max_value = float("-inf")

            # generate all possible actions for state s
            actions = generate_actions(state, self.K, self.N)

            # generate all posible next states
            possible_states = possible_items(state, self.K)

            for recommendations in actions:

                # count relevant items
                relevance = np.sum(self.U[state, recommendations] > self.u_min)

                if relevance == self.N:
                    # expected return if user selects one the recommended items
                    user_rec = (1/self.N) * sum((get_reward(next_state, self.cached) + self.gamma * V[next_state]) for next_state in recommendations)

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    action_value = (1-self.q) * (self.a*user_rec + (1-self.a)*user_not_rec)

                else :
                    # expected return if user selects one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    action_value = (1-self.q) * user_not_rec

                # update max action value
                if action_value > max_value:
                    max_value = action_value
                    best_action = recommendations

            # update new policy for the current state
            new_policy[state] = best_action

        return new_policy



    def Policy_Iteration_Recommender(self):
        # Initial policy
        self.policy = random_policy(self.K, self.N)
        self.all_policies.append(self.policy)

        # Initial value state vector
        V = np.zeros(self.K)

        # total costs, for plotting
        initial_cost, _ = calculate_cost(self.policy, self.N, self.U, self.cached)
        self.total_costs = [initial_cost]

        # convergence variable
        policy_stable = False

        # Perform policy iteration
        iteration = 1
        while not policy_stable:
            V_prev = np.copy(V)
            policy_prev = self.policy
            V = self.PolicyEvaluation(self.policy, max_iterations=1000, theta=self.theta)
            self.policy = self.PolicyImprovement(V)
            policy_cost, _ = calculate_cost(self.policy, self.N, self.U, self.cached)
            self.all_policies.append(self.policy)
            self.total_costs.append(policy_cost)


            # print(f'New policy, cost {calculate_cost(self.policy, self.U, self.cached)} : ')

            # print(self.policy)

            if policy_prev == self.policy:
                policy_stable = True
            iteration += 1


        # self.total_costs = [x for x in self.total_costs]
        self.iterations = iteration



    def plot(self):
        #  Plot total cost per iteration
        # print(self.total_costs)
        plt.plot(self.total_costs)
        plt.xlabel('Iteration')
        plt.ylabel('Total Cost')
        plt.title('Total Cost per Iteration')
        plt.show()
        return None


    def log(self):
        tb = pt()
        tb.title = 'Policy Iteration Recommender Policy'
        tb.field_names = ["State","Action", "Relevance"]

        for key, value in self.policy.items():
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

        fig, ax = plt.subplots()
        data = ax.imshow(colors, cmap="Paired_r", origin="lower", vmin=0)
        # plt.title('Cached Recommendations in Policy Iteration')
        plt.xlabel('Number of Iterations')
        plt.ylabel('States')
        ax.set_xticks(np.arange(i+1)-0.5, minor=True)
        ax.set_yticks(np.arange(self.K+1)-0.5, minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", size=0)
        # plt.legend(['Cached','Uncached'])
        # plt.colorbar(data)
        plt.show()


    def main(self):
        self.Policy_Iteration_Recommender()
        self.log()
        print('\n\n\n')
        plots.plot_policy_evaluation_heatmap(self.policy, self.U, self.u_min, self.cached, self.N, 'Policy Iteration')