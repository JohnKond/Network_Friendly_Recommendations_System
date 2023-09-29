# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from utils import generate_actions, possible_items, get_reward, calculate_cost, random_policy
import plots


class Policy_Iteration_Recommender:
    def __init__(self, env, gamma, theta):
        """
        Initialize the Policy Iteration Recommender.

        Parameters:
        - env: The environment.
        - gamma: The discount factor.
        - theta: The threshold for convergence.
        """

        self.env = env

        # discount factor
        self.gamma = gamma

        # terminating number ~0
        self.theta = theta

        # probability of choosing one of the rest k items
        self.p_k = 1/(self.env.K-1)

        # total costs of the algorithm per iteration
        self.total_costs = []

        # initialize policy
        self.policy = dict()

        self.all_policies = []

        self.iterations = 0

        # start procedure
        self.main()



    def PolicyEvaluation(self, policy, max_iterations):
        """
        Perform policy evaluation.

        Parameters:
        - policy: The policy to evaluate.
        - max_iterations: Maximum number of iterations for convergence.

        Returns:
        - V: The value state function.
        """

        # Initialize the value state function
        V = np.zeros(self.env.K)

        for i in range(max_iterations):

            V_prev = np.copy(V)

            # iterate over all possible states / items
            for state in range(self.env.K):

                recommendations = policy[state]
                
                relevance = np.sum(self.env.U[state, recommendations] > self.env.u_min)

                # generate all posible next states
                possible_states = possible_items(state, self.env.K)

                if relevance == self.env.N:

                    # expected return if user selects one the recommended items
                    # user_rec = (1/self.N) * ((get_reward(rec_item1, self.cached) + self.gamma * V[rec_item1]) + (get_reward(rec_item2, self.cached) + self.gamma * V[rec_item2]))

                    user_rec = (1/self.env.N) * sum((get_reward(next_state, self.env.cached) + self.gamma * V[next_state]) for next_state in recommendations)

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.env.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    V[state] = (1-self.env.q) * (self.env.alpha*user_rec + (1-self.env.alpha)*user_not_rec)

                else :

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.env.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    V[state] = (1-self.env.q) * user_not_rec



            # convergence check
            if np.max(np.abs(V - V_prev)) <= self.theta:
                break

            # max_iterations check
            if i == max_iterations :
                print('Policy Evaluation : Max iterations reached')
                break
        return V



    def PolicyImprovement(self, V):
        """
        Perform policy improvement.

        Parameters:
        - V: The value state function.

        Returns:
        - new_policy: The updated policy.
        """

        #initialize new policy dict
        new_policy = {}

        # iterate over all possible states / items
        for state in range(self.env.K):

            # for max comparison
            max_value = float("-inf")

            # generate all possible actions for state s
            actions = generate_actions(state, self.env.K, self.env.N)

            # generate all posible next states
            possible_states = possible_items(state, self.env.K)

            for recommendations in actions:

                # count relevant items
                relevance = np.sum(self.env.U[state, recommendations] > self.env.u_min)

                if relevance == self.env.N:
                    # expected return if user selects one the recommended items
                    user_rec = (1/self.env.N) * sum((get_reward(next_state, self.env.cached) + self.gamma * V[next_state]) for next_state in recommendations)

                    # expected return if user does NOT select one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.env.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    action_value = (1-self.env.q) * (self.env.alpha*user_rec + (1-self.env.alpha)*user_not_rec)

                else :
                    # expected return if user selects one the recommended items
                    user_not_rec = self.p_k * sum((get_reward(next_state, self.env.cached) + self.gamma *V[next_state]) for next_state in possible_states)
                    action_value = (1-self.env.q) * user_not_rec

                # update max action value
                if action_value > max_value:
                    max_value = action_value
                    best_action = recommendations

            # update new policy for the current state
            new_policy[state] = best_action

        return new_policy



    def Policy_Iteration_Recommender(self):
        """
        Perform policy iteration for the recommender system.
        """
         
        # Initial policy
        self.policy = random_policy(self.env.K, self.env.N)
        self.all_policies.append(self.policy)

        # Initial value state vector
        V = np.zeros(self.env.K)

        # total costs, for plotting
        initial_cost, _ = calculate_cost(self.policy, self.env.N, self.env.U, self.env.cached)
        self.total_costs = [initial_cost]

        # convergence variable
        policy_stable = False

        # Perform policy iteration
        iteration = 1
        while not policy_stable:
            V_prev = np.copy(V)
            policy_prev = self.policy
            V = self.PolicyEvaluation(self.policy, max_iterations=1000)
            self.policy = self.PolicyImprovement(V)
            policy_cost, _ = calculate_cost(self.policy, self.env.N, self.env.U, self.env.cached)
            self.all_policies.append(self.policy)
            self.total_costs.append(policy_cost)

            if policy_prev == self.policy:
                policy_stable = True
            iteration += 1

        self.iterations = iteration


    def log(self):
        """
        Print the policy details.
        """
        tb = pt()
        tb.title = 'Policy Iteration Recommender Policy'
        tb.field_names = ["State","Action", "Relevance"]

        for key, value in self.policy.items():
            tb.add_row([key,value, f'{self.env.U[key,value]}'])
        print(tb)


    def main(self):
        """
        Main function to execute the policy iteration recommender.
        """
        self.Policy_Iteration_Recommender()
        self.log()
        print('\n\n\n')
        plots.plot_policy_evaluation_heatmap(self.policy, self.env.U, self.env.u_min, self.env.cached, self.env.N, 'Policy Iteration')