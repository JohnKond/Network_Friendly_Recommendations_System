#################################################
#            RL PROJECT - PHASE 1               #
#       Friendly Recomendation System           #
#           Ioannis Kontogiorgakis              #
#################################################

import numpy as np

# number of items/videos
K = 50  

# relevance threshold in [0,1]
u_min = 0.5

# number of cached items
C = 0.2 * K


# number of recomendations the user gets
# after watching a content
N = 2


# Relation matrix
U = np.random.random((K, K))
np.fill_diagonal(U, 0)




print(U)