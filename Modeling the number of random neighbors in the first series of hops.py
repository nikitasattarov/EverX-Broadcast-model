# An algorithm for modeling the smallest number of random neighbors
# from the number of validators in the first series of hops.
# So that p percent of validators are covered by a block candidate.
# Each validator remembers who he sent the block to and from whom the block came to him.
# Version without parallel computing.
# It was launched using Python 3 in The Jupyter Notebook.

import numpy as np
import copy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 

# Draw a graph of the simulated function n(N).
def Plot_n_N(simul_n_dict):
    values_x = simul_n_dict.keys()
    values_y = simul_n_dict.values()
    plt.plot(values_x, values_y, 'bo', label="simulation", ms = 5) 
    plt.xlabel('N') 
    plt.ylabel('n') 
    plt.legend(loc = 'best', fancybox = True, shadow = True) 
    plt.grid(True) 
    plt.show()

    
# Initialize incoming variables.
exp_number = 500 # Number of simulation iterations
signf_lvl = 99 # The required percentage of coverage of validators by the block candidate
N_start = 100 # Initial value of the number of validators
N_finish = 400 # Final value of the number of validators
Delta_N = 30 # Step of the value of the number of validators
num_of_hops = 3 # Number of hops
n_init = 2 # Initial value of the number of random neighbors


# Keys are the numbers of validators.
# Values are simulated numbers of random neighbors.
simul_n_dict = {}

# Consider all values of N from a given range with a given step
for N in tqdm(range(N_start, N_finish + 1, Delta_N)):
    
    # Initialize a dictionary of possible random neighbors for each validator.
    # Keys are validator indexes.
    # Values are a list of indexes of possible random neighbors.
    # Indexes in the range from 1 to N.
    # For each validator, we remove the index of itself.
    dict_random_neighbors = {i: [j for j in range(1, N + 1)] for i in range(1, N + 1)}
    for i in range(1, N + 1):
        dict_random_neighbors[i].pop(i - 1)
    
    # We start the modeling algorithm.
    # We take the current value of n as the initial value.
    # We use a loop to make it easier to track the running time of the algorithm using tqdm.notebook.
    # Next, a node is a validator.
    for n in tqdm(range(n_init, N, 1)):
        
        # The total coverage for all iterations of the loop for the current value of n.
        total_cov = 0
        
        for cur_exp_number in tqdm(range(exp_number)):
            
            # Dictionary of avoided neighbors.
            # Keys are validator indexes.
            # Values are a sets of neighbors from whom we have already received a block
            # and to whom we have already sent a block.
            avoided_neighbors = {i: set() for i in range(1, N + 1)}
            
            # Initialization of a set of covered nodes. Always the starting node is a node with index 1.
            covered_nodes = {1}
            
            # Initialization of the set of those nodes that received a block in the current hop.
            new_nodes = {1}
            
            for cur_hop in range(num_of_hops):
                
                # Initialization of the set of nodes that will forward the block.
                init_nodes = copy.deepcopy(new_nodes)
                
                # Reset new_nodes.
                new_nodes = set()
                
                for init_node in init_nodes:
                    
                    # Initialize the list of random neighbors to whom the validator
                    # can send a block on the current hop.
                    random_choice_list = list(set(dict_random_neighbors[init_node]) - set(avoided_neighbors[init_node]))
                    
                    # If the list of random neighbors is not empty, then run.
                    if len(random_choice_list) > 0:
                        
                        # If the validator has the number of possible random neighbors less than n,
                        # then we send it to the maximum possible number of random neighbors.
                        cur_size = min(n, len(random_choice_list))
                        
                        # Generating indexes of random neighbors.
                        random_nodes = np.random.choice(random_choice_list, size=cur_size, replace=False, p=None)
                    else:
                        random_nodes = []

                    # To each random neighbor, we add the node from which the block came
                    # to the set of avoided neighbors. 
                    for cur_rand_node in random_nodes:
                        avoided_neighbors[cur_rand_node].add(init_node)
                    
                    # For the sending node, we add to the set of avoided neighbors those nodes
                    # to which it sent the block.
                    avoided_neighbors[init_node] = avoided_neighbors[init_node].union(random_nodes)
                    
                    # Update the set of those nodes that received a block in the current hop.
                    new_nodes = new_nodes.union(random_nodes)
                
                # Update the set of covered nodes.
                covered_nodes = covered_nodes.union(new_nodes)
            
            # Update the value of the total coverage.
            total_cov = total_cov + len(covered_nodes)

        # If the total coverage turned out to be greater than or equal to
        # the required total coverage (the required coverage multiplied by the number of iterations), 
        # then we stop the loop (stop increasing the variable n).    
        if (100 * total_cov >= N * signf_lvl * exp_number):
            break
        
    # Write the simulated value of n into the dictionary.                                
    simul_n_dict[N] = n
    
    # For the next value of N, we set the initial value of the variable n_init
    # as the current simulated value of n, since the function n(N) is increasing.
    n_init = n


    
Plot_n_N(simul_n_dict)
#print(simul_n_dict)
