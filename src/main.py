#########################################################
#                   RL PROJECT                          #
#       Network Friendly Recomendation System           #
#           Ioannis Kontogiorgakis                      #
#########################################################

import numpy as np
import random
import torch
from PolicyIteration import Policy_Iteration_Recommender
from DQN import DQN_run
from utils import generate_relevance_matrix_cached


seed_no = 67
np.random.seed(seed_no)
# random.seed(seed_no)

if __name__ == '__main__':
    # if GPU is to be used
    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # print(np.random.choice(10, 2, replace=False))
    U, cached = generate_relevance_matrix_cached(10,False)
    print('Cached items :',sorted(cached))

    policy_iteration_recommender = Policy_Iteration_Recommender(
        K = 10, U = U, cached=cached, N = 2, u_min = 0.5, q = 0.2, a = 0.8, gamma = 0.99, theta = 0.001, symmetric = False)
    
    # policy_iteration_recommender.log()
    # policy_iteration_recommender.plot_grid()


    # RUN DQN
    cost_per_episode_K_10, reward_per_episode_K_10 = DQN_run(K=10,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=500, device=device)
    # cost_per_episode_K_20, reward_per_episode_K_20 = DQN_run(K=20,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_50, reward_per_episode_K_50 = DQN_run(K=50,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_100, reward_per_episode_K_100 = DQN_run(K=100,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
    # cost_per_episode_K_1000, reward_per_episode_K_1000 = DQN_run(K=1000,N=2,u_min=0.5, alpha=0.8, q=0.2, gamma=0.99, num_episodes=6000)
