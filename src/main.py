#########################################################
#       Network Friendly Recomendation System           #
#       Author :  Ioannis Kontogiorgakis                #
#########################################################

import numpy as np
import random
import torch
from Environment import RecommendationEnvironment
from PolicyIteration import Policy_Iteration_Recommender
from QLearning import Q_Learning_Recommender
import experiments
from DQN import DQN_run
import argparse


seed_no = 105
np.random.seed(seed_no)
random.seed(seed_no)


parser = argparse.ArgumentParser(description='Recommender System')
# Algorithm choice
parser.add_argument('--alg', choices=['PI', 'QL', 'DQN'], required=True,
                    help='Algorithm to run: PI (Policy Iteration), QL (Q-Learning), DQN (Deep Q-Network)')

# Total number of items
parser.add_argument('--K', type=int, default=10,
                    help='Total number of items (positive integer)')

# Total number of recommendations
parser.add_argument('--N', type=int, default=2,
                    help='Total number of recommendations (positive integer)')

# Relevant threshold
parser.add_argument('--u_min', type=float, default=0.3, 
                    help='Relevant threshold (float from 0 to 1)')

# Probability of user quit
parser.add_argument('--q', type=float, default=0.2, 
                    help='Probability of user quitting after watching an item (float from 0 to 1)')

# Probability of user choosing to watch a recommended item if all recommendations are relevant
parser.add_argument('--alpha', type=float, default=0.2, 
                    help='Probability of user choosing to watch a recommended item if all recommendations are relevant (float from 0 to 1)')

parser.add_argument('--num_episodes', type=int, default=5000,
                    help='Total number of episodes for Q-Learning and DQN.(positive integer)')

args = parser.parse_args()

print('Selected Algorithm:', args.alg)
print('Total Number of Items:', args.K)
print('Total Number of Recommendations:', args.N)
print('Relevant Threshold:', args.u_min)
print('Probability of User Quit:', args.q)
print('Probability of User Choosing Recommended Item:', args.alpha)
print('Total number of episodes :',args.num_episodes)





if __name__ == '__main__':

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the environment
    env = RecommendationEnvironment(K=args.K, N=args.N, u_min=args.u_min, q=args.q, alpha=args.alpha)


    if args.alg == 'PI':
        policy_iteration_recommender = Policy_Iteration_Recommender(
            env = env, gamma = 0.99, theta=0.001)
    elif args.alg == 'QL':
        q_learning_recommender = Q_Learning_Recommender(
            env=env, gamma = 0.99, lr=0.001, num_episodes=args.num_episodes)    
    elif args.alg == 'DQN':
        DQN_run(
        env=env, gamma=0.99, num_episodes = args.num_episodes, device = device)



    # Experiments 

    # Run Q-Learning for different number of items (K)
    # experiments.q_learning_multiple_K()

    # Run Q-Learning for different u parameters
    # experiments.q_learning_multiple_u()

    # Run Q-Learning for different q parameters
    # experiments.q_learning_multiple_q()

    # Run Q-Learning for different alpha parameters
    # experiments.q_learning_multiple_a()

    # Run Q-Learning for different number of recommendations
    # experiments.q_learning_multiple_N()

    # Run DQN for different number of items (K)
    # experiments.DQN_multiple_K(device)

    # Run DQN for different u parameters
    # experiments.DQN_multiple_u(device)

    # Run DQN for different q parameters
    # experiments.DQN_multiple_q(device)

    # Run DQN for different alpha parameters
    # experiments.DQN_multiple_a(device)

    # Run DQN for different N
    # experiments.DQN_multiple_N(device)

    # Run DQN for large environment
    # experiments.DQN_multiple_K_large(device)

    # compare DQN and QL
    # experiments.QL_DQN_compare(device)

    print('End of execution')