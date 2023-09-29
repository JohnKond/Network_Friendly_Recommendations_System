# Network Friendly Recommendation System
## Author : Ioannis Kontogiorgakis


![Recommendation System overview](https://github.com/JohnKond/Network_Friendly_Recommendations_System/blob/main/environment_draw.png?raw=true)
## Abstract


The objective of this project is to create an intelligent system that optimizes content recommendations while
minimizing the impact on network resources. By modeling the recommendation problem as a Markov Decision Process
(MDP), we can analyze and optimize the system’s behavior by considering user preferences, network constraints, and
content relevance. Through the integration of reinforcement learning and network optimization techniques, we seek tocreate a robust and efficient recommendation system that adapts to individual user preferences and maximizes the utilization of available network resources. We will attempt to model the system’s decision making using 3 Reinforcement Learning algorithms:

- Policy Iteration
- Q-Learning
- Deep Q-Networks with Experience Replay

and we will analyze their interaction with respect to the system and the user. For more information, please see the project report.


## Project Structure

The project is organized as follows:

- `src/DQN.py`: This file contains the source implementation code of the Deep Q-Network algorithm with experience replay.
- `src/PolicyIteration.py`: This file contains the source implementation code of the Policy Iteration algorithm.
- `src/QLearning.py`: This file contains the source implementation code of the Q-Learning algorithm.
- `src/plots.py`: This file contains the code for the plots generated in this project.
- `src/experiments.py`: This file contains the functions for all the experiments conducted in the report.
- `src/main.py`: This file contains the main execution script for this project.
- `requirements.txt`: This file lists all the required Python packages and their versions needed to run the project.


## Installation

To run the code, follow these steps:

1. Install the required packages by running the command :  pip install -r /path/to/requirements.txt
2. Navigate to the `src/` directory in your terminal or command prompt.
3. The script has the following parameters:
    - **--alg** (algorithm to run)
    - **--K** (number of items)
    - **--N** (number of recommendations)
    - **--u_min** (relevance threshhold)
    - **--q** (probability of the user to quit the session)
    - **--alpha** (probability of the user to select a recommended item)
    
Please read carefully the project report in order to better understand the impact of the above parameters.

4. Run the `main.py` script by executing the following command and using the your preferred parameters: 
```python main.py --alg DQN --K 1000 --N 6 --u_min 0.3 --q 0.2 --alpha 0.8 --num_episodes 1000```
