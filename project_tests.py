
from project_methods import *

def test1( _h_graph):
    l_h_graph = list( _h_graph)
    row = l_h_graph[0]
    gen = combinations(row, 2)
    for tup in gen:
        fs = frozenset(tup)
        listed = list(fs)
        print("tuple:", tup, "fs:", fs, "list:", listed, "types:") #, end='' )
        for i in listed:
            print(type(i)) #, end='')
        # print()
        # for element in tup:
        #     z = 1
    return
def test2( hypergraph, G_orig):
    _adj_G_orig      = get_adjacency_from_Graph( G_orig)
    _adj_G_projected = get_adjacency_from_hypergraph( hypergraph)
    output = get_Prob_G_if_H( _adj_G_projected, _adj_G_orig)
    print("Adjacent?", output)
    #_adj_G_projected = get_adjacency_from_hypergraph( hypergraph)
    return

### Main ##########
print("--Tests script--")
FILENAME = "../windsurfers.gt"
G_orig   = init( FILENAME )
if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
print("Loaded file", FILENAME, "\nrun tests")
h_graph  = init_hypergraph(G_orig)
