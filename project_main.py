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

FILENAME = "../source_data/windsurfers.gt"
G_orig   = init( FILENAME )
if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
print("Loaded file", FILENAME, "\nrunning algorithm...")
try:
    result = find_best_hypergraph( G_orig)
except:
    print("Error.")

# Visualisation

stop_execution = input("Algorithm finished. Press any key to exit...")
