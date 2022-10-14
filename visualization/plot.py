import matplotlib.pyplot as plt
import networkx as nx
import ast
import numpy as np

length = 4
width = 5
test_size = 0.5
shadow_size = 50
data_name = 'orig'
d = 1
q1 = 0
q2 = 0

# note to self: only do real plots for test_size = 0.5 and shadow_size = 1000
# test file
#f = open('./../results/orig_data_clf_coefficients/results_4x5_orig_data_test_size=0.5_shadow_size=1000.txt', 'r')
f = open('./../clean_results/new_algorithm/test_size={}_shadow_size={}_{}_data_qubits_d={}/coefficients_{}x{}.txt'.format(test_size, shadow_size, data_name, d, length, width), 'r')
for line in f:
    if line[0:1] == '(':
        q1, q2 = ast.literal_eval(line[11:-1])
    else:
        edge_and_coef = ast.literal_eval(line)
        coef = np.array(list(map(lambda x : x[1], edge_and_coef)))

        G = nx.grid_2d_graph(length, width)

        pos = {(x, y) : (y, -x) for x, y in G.nodes()}
        node_to_int = {}
        for i in range(len(G.nodes())):
            (x, y) = list(G.nodes())[i]
            node_to_int[(x, y)] = i + 1

        edge_cmap = plt.cm.viridis
        line_width = coef * 100
        edge_color = coef * 100
        vmin = min(edge_color)
        vmax = max(edge_color)

        # this is same order as all_edges in algorithm
        edges = list(map(lambda e : (node_to_int[e[0]], node_to_int[e[1]]), list(G.edges())))

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color='#ffffff',
            edgecolors='#000000',
            linewidths=3.5,
            node_size=500)
        nx.draw_networkx_edges(
            G,
            pos,
            edge_cmap=edge_cmap,
            width=line_width,
            edge_color=edge_color,
            edge_vmin=vmin,
            edge_vmax=vmax)

        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_to_int,
            font_family='avenir',
            verticalalignment='center_baseline')

        # creating colorbar
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=vmin/100, vmax=vmax/100))
        sm._A = []
        plt.colorbar(sm)

        ax = plt.gca()
        ax.set_axis_off()
        ax.margins(0.10)

        plt.title('Coefficients of ML Model for (q1, q2) = ({}, {})'.format(q1, q2), fontname='avenir')
        plt.savefig('./plots/test_plots_test_size={}_shadow_size={}_{}_data_qubits_d={}_{}x{}/q1={}_q2={}.png'.format(test_size, shadow_size, data_name, d, length, width, q1, q2))
        plt.clf()
        #plt.show()

f.close()