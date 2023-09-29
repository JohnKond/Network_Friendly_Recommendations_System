import numpy as np
import plots
from QLearning import Q_Learning_Recommender
from DQN import DQN_run
from utils import generate_relevance_matrix_cached
from Environment import RecommendationEnvironment


def q_learning_multiple_u():
    """
    Run Q-Learning experiments varying the u_min parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    q = 0.2
    alpha = 0.8
    gamma = 0.99
    lr = 0.01
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=0.1, q=q, alpha=alpha)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=0.5, q=q, alpha=alpha)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=0.9, q=q, alpha=alpha)


    ql_K_10_N_2_u_01 = Q_Learning_Recommender(
        env=env_U_01, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_u_01_reward = ql_K_10_N_2_u_01.reward_per_ep
    print('Total sum u = 0.1: ', np.sum(ql_K_10_N_2_u_01_reward) / num_episodes)


    ql_K_10_N_2_u_05 = Q_Learning_Recommender(
        env=env_U_05, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_u_05_reward = ql_K_10_N_2_u_05.reward_per_ep
    print('Total sum u = 0.5: ', np.sum(ql_K_10_N_2_u_05_reward) / num_episodes)
    
    ql_K_10_N_2_u_09 = Q_Learning_Recommender(
        env=env_U_09, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_u_09_reward = ql_K_10_N_2_u_09.reward_per_ep
    print('Total sum u = 0.9: ', np.sum(ql_K_10_N_2_u_09_reward) / num_episodes)

    
    reward_per_episode_list = np.array([ql_K_10_N_2_u_01_reward, ql_K_10_N_2_u_05_reward, ql_K_10_N_2_u_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['u = 0.1','u = 0.5','u = 0.9'],
                                       alg = 'Q-Learning')
    

def q_learning_multiple_q():
    """
    Run Q-Learning experiments varying the q parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    u_min = 0.3
    alpha = 0.8
    gamma = 0.99
    lr = 0.01
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.1, alpha=alpha)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.5, alpha=alpha)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.9, alpha=alpha)


    ql_K_10_N_2_q_01 = Q_Learning_Recommender(
        env=env_U_01, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_q_01_reward = ql_K_10_N_2_q_01.reward_per_ep
    print('Total sum q = 0.1: ', np.sum(ql_K_10_N_2_q_01_reward) / num_episodes)

    ql_K_10_N_2_q_05 = Q_Learning_Recommender(
        env=env_U_05, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_q_05_reward = ql_K_10_N_2_q_05.reward_per_ep
    print('Total sum q = 0.5: ', np.sum(ql_K_10_N_2_q_05_reward) / num_episodes)
    
    ql_K_10_N_2_q_09 = Q_Learning_Recommender(
        env=env_U_09, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_q_09_reward = ql_K_10_N_2_q_09.reward_per_ep
    print('Total sum q = 0.9: ', np.sum(ql_K_10_N_2_q_09_reward) / num_episodes)
    
    reward_per_episode_list = np.array([ql_K_10_N_2_q_01_reward, ql_K_10_N_2_q_05_reward, ql_K_10_N_2_q_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['q = 0.1','q = 0.5','q = 0.9'],
                                       alg = 'Q-Learning')


def q_learning_multiple_a():
    """
    Run Q-Learning experiments varying the alpha parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    u_min = 0.3
    q = 0.2
    gamma = 0.99
    lr = 0.01
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.1)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.5)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.9)


    ql_K_10_N_2_a_01 = Q_Learning_Recommender(
        env=env_U_01, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_a_01_reward = ql_K_10_N_2_a_01.reward_per_ep
    print('Total sum a = 0.1: ', np.sum(ql_K_10_N_2_a_01_reward) / num_episodes)


    ql_K_10_N_2_a_05 = Q_Learning_Recommender(
        env=env_U_05, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_a_05_reward = ql_K_10_N_2_a_05.reward_per_ep
    print('Total sum a = 0.5: ', np.sum(ql_K_10_N_2_a_05_reward) / num_episodes)
    
    ql_K_10_N_2_a_09 = Q_Learning_Recommender(
        env=env_U_09, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_a_09_reward = ql_K_10_N_2_a_09.reward_per_ep
    print('Total sum a = 0.9: ', np.sum(ql_K_10_N_2_a_09_reward) / num_episodes)
    
    reward_per_episode_list = np.array([ql_K_10_N_2_a_01_reward, ql_K_10_N_2_a_05_reward, ql_K_10_N_2_a_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['alpha = 0.1','alpha = 0.5','alpha = 0.9'],
                                       alg = 'Q-Learning')



def q_learning_multiple_K():
    """
    Run Q-Learning experiments varying the total number of content items (K).
    """
    # define stable parameters for the experiment
    N = 2
    q = 0.2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    lr = 0.001
    num_episodes = 10000
        
    env_K_10 = RecommendationEnvironment(K=10, N=N, u_min=u_min, q=q, alpha=alpha)
    env_K_50 = RecommendationEnvironment(K=50, N=N, u_min=u_min, q=q, alpha=alpha)
    env_K_80 = RecommendationEnvironment(K=80, N=N, u_min=u_min, q=q, alpha=alpha)


    ql_K_10_N_2 = Q_Learning_Recommender(
        env = env_K_10, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_10_N_2_reward = ql_K_10_N_2.reward_per_ep
    print('Total sum K = 10: ', np.sum(ql_K_10_N_2_reward) / num_episodes)


    ql_K_50_N_2 = Q_Learning_Recommender(
        env = env_K_50, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_50_N_2_reward = ql_K_50_N_2.reward_per_ep
    print('Total sum K = 50: ', np.sum(ql_K_50_N_2_reward) / num_episodes)
    
    ql_K_80_N_2 = Q_Learning_Recommender(
        env = env_K_80, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_80_N_2_reward = ql_K_80_N_2.reward_per_ep
    print('Total sum K = 80: ', np.sum(ql_K_80_N_2_reward) / num_episodes)
    
    reward_per_episode_list = np.array([ql_K_10_N_2_reward, ql_K_50_N_2_reward, ql_K_80_N_2_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['K=10','K=50','K=80'],
                                       alg='Q-Learning')
    

def q_learning_multiple_N():
    """
    Run Q-Learning experiments varying the number of recommended items (N).
    """
    # define stable parameters for the experiment
    K = 50
    q = 0.2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    lr = 0.001
    num_episodes = 10000
        
    env_N_2 = RecommendationEnvironment(K=K, N=2, u_min=u_min, q=q, alpha=alpha)
    env_N_4 = RecommendationEnvironment(K=K, N=4, u_min=u_min, q=q, alpha=alpha)
    env_N_6 = RecommendationEnvironment(K=K, N=6, u_min=u_min, q=q, alpha=alpha)


    ql_K_50_N_2 = Q_Learning_Recommender(
        env = env_N_2, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_50_N_2_reward = ql_K_50_N_2.reward_per_ep


    ql_K_50_N_4 = Q_Learning_Recommender(
        env = env_N_4, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_50_N_4_reward = ql_K_50_N_4.reward_per_ep
    
    ql_K_50_N_6 = Q_Learning_Recommender(
        env = env_N_6, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_50_N_6_reward = ql_K_50_N_6.reward_per_ep
    
    reward_per_episode_list = np.array([ql_K_50_N_2_reward, ql_K_50_N_4_reward, ql_K_50_N_6_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                    window_size=int(num_episodes/10),
                                    num_episodes=num_episodes,
                                    legend=['N=2','N=4','N=6'],
                                    alg='Q-Learning')



def DQN_multiple_u(device):
    """
    Run DQN experiments varying the u_min parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    q = 0.2
    alpha = 0.8
    gamma = 0.99
    lr = 0.01
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=0.1, q=q, alpha=alpha)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=0.5, q=q, alpha=alpha)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=0.9, q=q, alpha=alpha)


    _ , _, dqn_K_10_u_01_reward = DQN_run(
        env=env_U_01, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    print('Total sum u_min = 0.1: ', np.sum(dqn_K_10_u_01_reward))


    _ , _, dqn_K_10_u_05_reward = DQN_run(
        env=env_U_05, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum u_min = 0.5: ', np.sum(dqn_K_10_u_05_reward))
    
    
    _ , _, dqn_K_10_u_09_reward = DQN_run(
        env=env_U_09, gamma = gamma, num_episodes = num_episodes, device=device) 
    print('Total sum u_min = 0.9: ', np.sum(dqn_K_10_u_09_reward))

    
    reward_per_episode_list = np.array([dqn_K_10_u_01_reward, dqn_K_10_u_05_reward, dqn_K_10_u_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['u = 0.1','u = 0.5','u = 0.9'],
                                       alg = 'DQN')


def DQN_multiple_q(device):
    """
    Run DQN experiments varying the q parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    lr = 0.01
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.1, alpha=alpha)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.5, alpha=alpha)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=0.9, alpha=alpha)


    _ , _, dqn_K_10_q_01_reward = DQN_run(
        env=env_U_01, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    print('Total sum q = 0.1: ', np.sum(dqn_K_10_q_01_reward) / num_episodes)


    _ , _, dqn_K_10_q_05_reward = DQN_run(
        env=env_U_05, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum q = 0.5: ', np.sum(dqn_K_10_q_05_reward)/ num_episodes)
    
    
    _ , _, dqn_K_10_q_09_reward = DQN_run(
        env=env_U_09, gamma = gamma, num_episodes = num_episodes, device=device) 
    print('Total sum q = 0.9: ', np.sum(dqn_K_10_q_09_reward)/ num_episodes)

    
    reward_per_episode_list = np.array([dqn_K_10_q_01_reward, dqn_K_10_q_05_reward, dqn_K_10_q_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['q = 0.1','q = 0.5','q = 0.9'],
                                       alg = 'DQN')
    



def DQN_multiple_a(device):
    """
    Run DQN experiments varying the alpha parameter.
    """
    # define stable parameters for the experiment
    K = 10
    N = 2
    q = 0.2
    u_min = 0.3
    gamma = 0.99
    num_episodes = 10000

    env_U_01 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.1)
    env_U_05 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.5)
    env_U_09 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=0.9)


    _ , _, dqn_K_10_a_01_reward = DQN_run(
        env=env_U_01, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum a = 0.1: ', np.sum(dqn_K_10_a_01_reward)/ num_episodes)


    _ , _, dqn_K_10_a_05_reward = DQN_run(
        env=env_U_05, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum a = 0.5: ', np.sum(dqn_K_10_a_05_reward)/ num_episodes)
    
    
    _ , _, dqn_K_10_a_09_reward = DQN_run(
        env=env_U_09, gamma = gamma, num_episodes = num_episodes, device=device) 
    print('Total sum a = 0.9: ', np.sum(dqn_K_10_a_09_reward)/ num_episodes)

    
    reward_per_episode_list = np.array([dqn_K_10_a_01_reward, dqn_K_10_a_05_reward, dqn_K_10_a_09_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['alpha = 0.1','alpha = 0.5','alpha = 0.9'],
                                       alg = 'DQN')




def DQN_multiple_K(device):
    """
    Run DQN experiments varying the total number of content items (K).

    """
    # define stable parameters for the experiment
    N = 2
    q = 0.2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    num_episodes = 10000


    env_K_10 = RecommendationEnvironment(K=10, N=N, u_min=u_min, q=q, alpha=alpha)
    env_K_50 = RecommendationEnvironment(K=50, N=N, u_min=u_min, q=q, alpha=alpha)
    env_K_80 = RecommendationEnvironment(K=80, N=N, u_min=u_min, q=q, alpha=alpha)
    
    
    _ , _, dqn_K_10_N_2_reward = DQN_run(
        env=env_K_10, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    print('Total sum K = 10: ', np.sum(dqn_K_10_N_2_reward) / num_episodes)


    _ , _, dqn_K_50_N_2_reward = DQN_run(
        env=env_K_50, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum K = 50: ', np.sum(dqn_K_50_N_2_reward) / num_episodes)
    
    
    _ , _, dqn_K_80_N_2_reward = DQN_run(
        env=env_K_80, gamma = gamma, num_episodes = num_episodes, device=device)    
    print('Total sum K = 80: ', np.sum(dqn_K_80_N_2_reward) / num_episodes)

    
    reward_per_episode_list = np.array([dqn_K_10_N_2_reward, dqn_K_50_N_2_reward, dqn_K_80_N_2_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['K=10','K=50','K=80'],
                                       alg='DQN')
    

def DQN_multiple_N(device):
    """
    Run DQN experiments varying the number of recommended items (N).
    """
    # define stable parameters for the experiment
    K= 50
    q = 0.2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    num_episodes = 10000
    
    
    env_K_100 = RecommendationEnvironment(K=K, N=2, u_min=u_min, q=q, alpha=alpha)
    env_K_500 = RecommendationEnvironment(K=K, N=4, u_min=u_min, q=q, alpha=alpha)
    env_K_1000 = RecommendationEnvironment(K=K, N=6, u_min=u_min, q=q, alpha=alpha)
    # env_K_2000 = RecommendationEnvironment(K=2000, N=200, u_min=u_min, q=q, alpha=alpha)

    
    
    _ , _, dqn_N_2_reward = DQN_run(
        env=env_K_100, gamma = gamma, num_episodes = num_episodes, device=device)    
    

    _ , _, dqn_N_4_reward = DQN_run(
        env=env_K_500, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    _ , _, dqn_N_6_reward = DQN_run(
        env=env_K_1000, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    reward_per_episode_list = np.array([dqn_N_2_reward, dqn_N_4_reward, dqn_N_6_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['N=2','N=4','N=6'],
                                       alg='DQN')




def DQN_multiple_K_large(device):
    """
    Run DQN experiments varying the total number of content items (K) for larger values.
    """
    # define stable parameters for the experiment
    q = 0.2
    alpha = 0.8
    u_min = 0.3
    gamma = 0.99
    num_episodes = 10000
    
    
    env_K_100 = RecommendationEnvironment(K=100, N=3, u_min=u_min, q=q, alpha=alpha)
    env_K_500 = RecommendationEnvironment(K=500, N=5, u_min=u_min, q=q, alpha=alpha)
    env_K_1000 = RecommendationEnvironment(K=1000, N=6, u_min=u_min, q=q, alpha=alpha)
    # env_K_2000 = RecommendationEnvironment(K=2000, N=200, u_min=u_min, q=q, alpha=alpha)

    
    
    _ , _, dqn_K_100_reward = DQN_run(
        env=env_K_100, gamma = gamma, num_episodes = num_episodes, device=device)    
    

    _ , _, dqn_K_500_reward = DQN_run(
        env=env_K_500, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    _ , _, dqn_K_1000_reward = DQN_run(
        env=env_K_1000, gamma = gamma, num_episodes = num_episodes, device=device)    
    
    reward_per_episode_list = np.array([dqn_K_100_reward, dqn_K_500_reward, dqn_K_1000_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['K=100, N=10','K=500, N=50','K=1000, N=100'],
                                       alg='DQN')
    

def QL_DQN_compare(device):
    """
    Compare Q-Learning and DQN for a specific configuration.
    """
    # define stable parameters for the experiment
    K = 100
    N = 3
    u_min = 0.3
    q = 0.2
    gamma = 0.99
    alpha = 0.8
    lr = 0.01
    num_episodes = 10000

    env_N_3 = RecommendationEnvironment(K=K, N=N, u_min=u_min, q=q, alpha=alpha)



    ql_K_100_N_3 = Q_Learning_Recommender(
        env=env_N_3, gamma = gamma, lr = lr, num_episodes = num_episodes)    
    ql_K_100_N_3_reward = ql_K_100_N_3.reward_per_ep

    _ , _, dqn_K_100_N_3_reward = DQN_run(
        env=env_N_3, gamma = gamma, num_episodes = num_episodes, device=device)    

    
    reward_per_episode_list = np.array([ql_K_100_N_3_reward, dqn_K_100_N_3_reward])
    plots.plot_multiple_average_reward(reward_per_episode_list=reward_per_episode_list,
                                       window_size=int(num_episodes/10),
                                       num_episodes=num_episodes,
                                       legend=['QL: N=3','DQN: N=3'],
                                       alg = '')