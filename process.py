# This file contains functions to post-process data.

import os
import numpy as np
import pandas as pd
import ast

def process_orig_algorithm_output():
    # used to translate k \in [0, (length * width)^2) to pairs of qubits (q1, q2)
    # filters into different files for d = 1, 2, 3 just as new algorithm outputs
    # note for d = 1, qubits are still in slightly different order
    width = 5

    for data_name in ['orig', 'new']:
        length_range = list(range(4, 8))
        if data_name == 'orig':
            length_range = list(range(4, 10))
        
        for length in length_range:
            for test_size in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for shadow_size in [50, 100, 250, 500, 1000]:
                    for d in [1, 2, 3]:

                        # grid of qubits
                        grid = np.array(range(1, length * width + 1)).reshape((length, width))

                        # generate all edges in grid in same order as Xfull
                        all_edges = []
                        for i in range(0, length):
                            for j in range(1, width + 1):
                                if i != length - 1:
                                    all_edges.append((width * i + j, width * (i + 1) + j))
                                if j != width:
                                    all_edges.append((width * i + j, width * i + j + 1))

                        def calc_distance(q1, q2):
                            # Given two qubits q1, q2 (1-indexed integers) in length x width grid
                            # Output l1 distance between q1 and q2 in grid

                            pos1 = np.array(np.where(grid == q1)).T[0]
                            pos2 = np.array(np.where(grid == q2)).T[0]

                            return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

                        def get_nearby_qubit_pairs(d):
                            # Given distance d > 0
                            # Output all pairs of qubits that are within distance d of each other
                            
                            if d == 1:
                                return all_edges
                            
                            qubit_pairs = []
                            for q1 in range(1, length * width + 1):
                                for q2 in range(1, length * width + 1):
                                    dist = calc_distance(q1, q2)
                                    pair = tuple(sorted((q1, q2)))
                                    if dist == d and pair not in qubit_pairs:
                                        qubit_pairs.append(pair)
                            
                            return qubit_pairs

                        orig_file = './clean_results/orig_algorithm/test_size={}_shadow_size={}_all_qubits/results_{}x{}_{}_data.txt'.format(test_size, shadow_size, length, width, data_name)
                        if os.path.exists(orig_file):
                            r = open(orig_file, 'r')
                            new_dir = './clean_results/orig_algorithm_processed/test_size={}_shadow_size={}_qubits_d={}'.format(test_size, shadow_size, d)
                            if not os.path.exists(new_dir):
                                os.makedirs(new_dir)

                            f = open('{}/results_{}x{}_{}_data.txt'.format(new_dir, length, width, data_name), 'w')

                            qubits = get_nearby_qubit_pairs(d)

                            q1 = 0
                            q2 = 0
                            for line in r:
                                if line[:4] == 'k = ':
                                    k = int(line[4:-1])
                                    q1 = k // (length * width) + 1
                                    q2 = k % (length * width) + 1
                                    if (q1, q2) in qubits:
                                        print('(q1, q2) = ({}, {})'.format(q1, q2), file=f)
                                else:
                                    if (q1, q2) in qubits:
                                        print(line, file=f, end='')

                            r.close()
                            f.close()
    print('done.')

def make_dataframe():
    # used to create pandas dataframe compiling all data together

    width = 5
    data = []

    for data_name in ['orig', 'new']:
        length_range = list(range(4, 8))
        if data_name == 'orig':
            length_range = list(range(4, 10))
        
        for length in length_range:
            for test_size in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for shadow_size in [50, 100, 250, 500, 1000]:
                    for d in [1, 2, 3]:
                        row = ['{}x{}'.format(length, width), d, data_name, test_size, shadow_size, 'Best Previous']
                        # do processing for previous algorithm
                        orig_alg_file = './clean_results/orig_algorithm_processed/test_size={}_shadow_size={}_qubits_d={}/results_{}x{}_{}_data.txt'.format(test_size, shadow_size, d, length, width, data_name)
                        new_alg_file = './clean_results/new_algorithm/test_size={}_shadow_size={}_qubits_d={}/results_{}x{}_{}_data.txt'.format(test_size, shadow_size, d, length, width, data_name)

                        # remove and statement if want all data for previous algorithm
                        if os.path.exists(orig_alg_file) and os.path.exists(new_alg_file):
                            r = open(orig_alg_file, 'r')
                            avg_pred_errors = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
                            method_counter = 1
                            n = 0
                            for line in r:
                                if line[0:3] != '(q1':
                                    avg_pred_errors[method_counter] += ast.literal_eval(line[15:])[1]
                                    method_counter += 1
                                    if method_counter > 6:
                                        method_counter = 1
                                        n += 1
                            avg_pred_errors = {k : v / n for k, v in avg_pred_errors.items()}
                            best_avg_pred_error = max(avg_pred_errors.values())
                            row.append(best_avg_pred_error)
                        data.append(row)

                        # do processing for new algorithm
                        row = ['{}x{}'.format(length, width), d, data_name, test_size, shadow_size, 'New']
                        new_alg_file = './clean_results/new_algorithm/test_size={}_shadow_size={}_qubits_d={}/results_{}x{}_{}_data.txt'.format(test_size, shadow_size, d, length, width, data_name)
                        if os.path.exists(new_alg_file):
                            r = open(new_alg_file, 'r')
                            avg_pred_error = 0
                            n = 0
                            for line in r:
                                if line[0:3] != '(q1':
                                    avg_pred_error += ast.literal_eval(line)[1]
                                    n += 1
                            avg_pred_error /= n
                            row.append(avg_pred_error)     
                        data.append(row)

    return pd.DataFrame(data, columns=['System Size', 'Distance', 'Data Set', 'Test Size', 'Shadow Size', 'Algorithm', 'Avg Prediction Error'])

if __name__ == '__main__':
    #process_orig_algorithm_output()
    df = make_dataframe()
    df.to_excel('./clean_results/results.xlsx')
    print('done.')
