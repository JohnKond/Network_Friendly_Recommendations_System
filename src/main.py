#########################################################
#                   RL PROJECT                          #
#       Network Friendly Recomendation System           #
#           Ioannis Kontogiorgakis                      #
#########################################################

import numpy as np
import random
import torch
from PolicyIteration import Policy_Iteration_Recommender
from QLearning import Q_Learning_Recommender
from DQN import DQN_run
from utils import generate_relevance_matrix_cached
import experiments
import plots


seed_no = 24
np.random.seed(seed_no)
random.seed(seed_no)

if __name__ == '__main__':
    # if GPU is to be used
    # print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # define number of items
    K = 10

    # define number of recommendations
    N = 2

    #define relevance threshhold
    u_min = 0.3

    # probability for the user to quit
    q = 0.2

    # probability for the user to choose recommended items
    a = 0.8 

    # generate relevance matrix and cached items
    U, cached = generate_relevance_matrix_cached(K,True)
    print('Cached items :',sorted(cached))
    # U = np.loadtxt('U_simple_10')
    print('Relevance matrix :', U)

    # plot relevance matrix heatmap
    # plots.relevance_heatmap(U,u_min)


    
    # Example runs (Policy Iteration, Q-Learning, DQN)

    policy_iteration_recommender = Policy_Iteration_Recommender(
        K = K, N = N, U = U, cached = cached, u_min = u_min, q = q, a = a, gamma = 0.99, theta = 0.001)
    

    q_learning_recommender = Q_Learning_Recommender(
        K = K, N = N, U = U, cached = cached, u_min = u_min, q = q, a = a, gamma = 0.99, lr = 0.1, num_episodes = 10000)    

    # DQN_run(
        # K=K, N=N, U=U, cached=cached, u_min=u_min, alpha=a, q=q, gamma=0.99, num_episodes = 10000, device = device)





    # Experiments 

    # Run Q-Learning for different u parameters
    # experiments.q_learning_multiple_u()

    # Run Q-Learning for different number of items (K)
    # experiments.q_learning_multiple_K()

    # Run DQN for small environment
    # experiments.DQN_multiple_K_small(device)

    # Run DQN for large environment
    # experiments.DQN_multiple_K_large(device)



    # RUN DQN
    # cost_per_episode_K_10, reward_per_episode_K_10 = DQN_run(K=10,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=500, device=device)
    # cost_per_episode_K_20, reward_per_episode_K_20 = DQN_run(K=20,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_50, reward_per_episode_K_50 = DQN_run(K=50,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_100, reward_per_episode_K_100 = DQN_run(K=100,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_1000, reward_per_episode_K_1000 = DQN_run(K=1000,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)


    print('End of execution')