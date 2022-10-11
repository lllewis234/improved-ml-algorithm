# used to translate k \in [0, (length * width)^2) to pairs of qubits (q1, q2)

width = 5
data_name = 'new'

for length in range(4, 8):
    with open('./results/orig_algorithm/orig_algorithm_test_size=0.4/orig_algorithm_{}_data_k/results_{}x{}_all_qubits.txt'.format(data_name, length, width), 'r') as r:
        with open('./results/orig_algorithm/orig_algorithm_test_size=0.4/orig_algorithm_{}_data/results_{}x{}_all_qubits.txt'.format(data_name, length, width), 'w') as f:
            for line in r:
                if line[:4] == 'k = ':
                    k = int(line[4:-1])
                    q1 = k // (length * width) + 1
                    q2 = k % (length * width) + 1
                    print('(q1, q2) = ({}, {})'.format(q1, q2), file=f)
                else:
                    print(line, file=f, end='')

print('done')
