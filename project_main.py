### source ######################################
# arXiv:2008.04948v3 [cs.SI] 25. May 2021
# Hypergraph reconstruction from network data
# J-G. Young, G. Petri, T. P. Peixoto

# https://docs.python.org/3/library/math.html
# https://graph-tool.skewed.de/static/doc/graph_tool.html
# math.comb(n,k)    # binomial coefficient
# math.factorial(x)

from project_methods import *

# run the algorithm

FILENAME = "../windsurfers.gt"
G_orig   = init( FILENAME )
if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
h_graph  = init_hypergraph(G_orig)
# run though h_graph candidates

### evidence, normalization
# add each iteration of P_G_H * P_H to this array, and then sum them up at the end
P_G_arr = []
placeholder_list = [ h_graph ]

N = len(list(G_orig.vertices())) # confirm this?
#
for i in placeholder_list:
    L     = min(10, N) # placeholder, max size of hyperedge, min 2

    P_G_H = get_Prob_G_if_H()
    P_H   = get_Prob_H( i, L, N)
    top   = P_G_H * P_H

    P_G_arr.append(top)

# Visualisation

stop_execution = input("Algorithm finished. Press any key to exit...")
