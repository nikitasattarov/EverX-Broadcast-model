# An algorithm for modeling AVM and delta functions
# from the number of validators and
# from the number of random neighbors.

# Version without parallel computing
# (running slowly, recommended exp_number values are no more than 1000).

# It was launched using Python 3 in The Jupyter Notebook.

import numpy as np
import math
import json
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# Draw a graph of the simulated function AVM(N) for current N.
def plot_AVM_m(dict_data_base, N):
    m = list(dict_data_base[N].keys())
    y_max_avm = list(zip(*dict_data_base[N].values()))[0]
    plt.plot(m, y_max_avm, 'o', ms = 3, label="simulation", color = 'blue') 
    plt.xlabel('m (the number of random neighbors during' + \
               '\n the transfer of signatures between participants)') 
    plt.ylabel('AVM')
    plt.legend(loc = 'best', fancybox = True, shadow = True) 
    plt.grid(True)
    plt.show()

    
# Draw a graph of the simulated function delta(N) for current N.   
def plot_delta_m(dict_data_base, N):
    m = list(dict_data_base[N].keys())
    y_min_delta = list(zip(*dict_data_base[N].values()))[1]
    plt.plot(m, y_min_delta, 'o', ms = 3, label="simulation", color = 'blue') 
    plt.xlabel('m (the number of random neighbors during' + \
               '\n the transfer of signatures between participants)') 
    plt.ylabel(r'$\delta$')
    plt.legend(loc = 'best', fancybox = True, shadow = True) 
    plt.grid(True)
    plt.show()


# Number of simulation iterations 
exp_number = 100

# List of the number of validators for which we are modeling
N_list = (100, 200)

# A dictionary of lists in which the "worst" modeled values (avm, delta)
# will be found in the dictionary values for each N, for each m. 
dict_data_base = {N: {m: list() for m in range(1, min(int(N / 2) + 2, 200))} \
                  for N in N_list}

# Run a loop for each value of N, for which we will model.
for N in tqdm(N_list):
    
    # Initialize a dictionary of possible random neighbors for each validator.
    # Keys are validator indexes.
    # Values are a list of indexes of possible random neighbors.
    # Indexes in the range from 1 to N.
    # For each validator, we remove the indem of itself.
    dict_random_neighbors = {i: [j for j in range(1, N + 1)] \
                             for i in range(1, N + 1)}
    for i in range(1, N + 1):
        dict_random_neighbors[i].pop(i - 1)
   
    # The value of collected signatures (block candidate statuses) required for
    # the validator to send all collected signatures to Masterchain.    
    am_col_sign = int(N / 2) + 1
    
    # Run a loop for each value of m from a given range.
    for m in tqdm(range(1, min(int(N / 2) + 2, 200))):
        
        for cur_exp_number in range(exp_number):    
            
            # Dictionary of collected signatures.
            # Keys are validator indexes.
            # Values are indexes of validators whose
            # signatures the current validator has.
            # Initially, each validator has its own signature.
            dict_collect_sign = {i: {i} for i in range(1, N + 1)}
            
            # A node is a validator.
            # Each node sends signatures.
            for cur_node in range(1, N + 1):
                
                # Generating a list of random neighborsand
                # and filling in dict_collect_sign dictionary
                random_nodes = np.random.choice(dict_random_neighbors[cur_node], \
                                                size=m, replace=False, p=None)
                for cur_send_node in random_nodes:
                    dict_collect_sign[cur_send_node].add(cur_node)
                    
            # The expected value of the normal distribution
            # (function of the number of validators who have collected the
            # corresponding number of signatures from the number of
            # these collected signatures)
            u = m + 1
            
            # A set of indexes of nodes that we are considering
            # for forwarding to the Masterchain.
            num_nodes = set(i for i in range(1, N + 1))
            
            # Changing the delta from the maximum.
            for delta in range(N - u, 0, -1):
                
                # The initial value of the number of validators that
                # send signatures to the Masterchain.
                avm = 0
                
                # A set of signature indexes that will come to
                # the Master Chain.
                master_sign = set()
                
                # Сheck each node.
                for cur_node in num_nodes:
                    
                    # A set of nodes that we will remove from 
                    # the set of checked nodes if this delta does not suit us.
                    
                    # So as not to double-check the extra nodes.
                    delete_nodes = set()
                    
                    # If current node has more than
                    # the required number of signatures, 
                    # then we send its signatures to the Masterchain.
                    if (u + delta <= len(dict_collect_sign[cur_node])):
                        avm = avm + 1
                        master_sign = master_sign.union(dict_collect_sign[cur_node])
                        
                        delete_nodes.add(cur_node)
                
                # If the required number of signatures has not been 
                # collected in the Masterchain,
                # then we record the data and stop the algorithm,
                # else we correct the set of nodes being checked and
                # reduce the delta.
                if (len(master_sign) >= am_col_sign):
                    dict_data_base[N][m].append((avm, delta))
                    break
                num_nodes = num_nodes - delete_nodes
        
# We get a dictionary, in each value of which there is
# a list of pairs (AVM, delta).

# Let's select all of these pairs of "worst case"
# (when the delta value is minimal) and overwrite the data.
for N in N_list:
    for m in range(1, min(int(N / 2) + 2, 200)):
        dict_data_base[N][m] = dict_data_base[N][m] \
        [np.argmin(list(zip(*dict_data_base[N][m]))[1])] 
        
#print(dict_data_base)

N_plot = 100
plot_AVM_m(dict_data_base, N_plot)
plot_delta_m(dict_data_base, N_plot)
