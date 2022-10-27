### source ######################################
# arXiv:2008.04948v3 [cs.SI] 25. May 2021
# Hypergraph reconstruction from network data
# J-G. Young, G. Petri, T. P. Peixoto

# https://docs.python.org/3/library/math.html
# https://graph-tool.skewed.de/static/doc/graph_tool.html
# math.comb(n,k)    # binomial coefficient
# math.factorial(x)

# necessary packages
from           math import *
#from graph_tool.all import *

### Find best hypergraph ########################

def init( _filename):
    try:
        _g = load_graph( _filename)
        return _g
    except:
        print("ERROR while loading", _filename)
        return None

def init_hypergraph( _G_orig):
    # Returns datatype: dict of frozensets of graph_tool.Vertices, maps to Z+
    _h_graph = dict()
    # # Takes the original graph _G_orig, adds all its edges to the h_graph
    # for edge in _G_orig.edges():
    #     src = _G_orig.vertex_index[edge.source()]
    #     trg = _G_orig.vertex_index[edge.target()]
    #     _h_graph[ frozenset( [src, trg] ) ] = 1
    ### Current methodology: Adds all maximal cliques as hyperedges to the hypergraph.
    # Adds every maximal clique in G_orig as hyperedges to h_graph
    _m_cliques = max_cliques( _G_orig )
    for clique in _m_cliques:
        fs = frozenset( clique )
        _h_graph[ fs ] = 1 # + _h_graph.get(fs, 0)
    ### Alternative methods include random init, edge init, or empty (page 6, last paragraph before section [D] )
    #
    return _h_graph

### evidence, normalization
# add each iteration of P_G_H * P_H to this array, and then sum them up at the end
P_G_arr = []

# projection component -- equation (2)
#   todo:
# flatten the hyperedges and compare if
# the resulting graph G matches with G_orig
def get_Prob_G_if_H( _hypergraph):
    # todo shenanigans here
    # take this h_graph and map all of its h_edges into pairwise interactions and store them in adj_G
    # take the original graph's edges, turn that into a list orig_adj_G, and compare the two
    # return 1 if they match and 0 if they don't

    # given a hyperedge E_h in h_graph, mark all nodes it connects as pairwise connected in G
    adj_G = dict()
    # -> how to iterate a python set?
    #   <- convert to list
    # ...
    return 1 # placeholder


# hypergraph prior -- equation (12)
def get_Prob_H( _hypergraph):
    L = 10 # max size of hyperedge, min 2
    foldproduct = 1
    for k in range(2, L + 1):
      N    = k # placeholder # size of original graph? size: number of edges
      #nu_k = 1 # placeholder
      E_
      E_k  = 1 # placeholder
      Z_k  = 1 # placeholder
      mu   = (L - 1)*log(1/(1-E/comb(N,2)))
               #nu_k * comb(N, k)
      #print("k is", k, mu)
      P_H_k  = ( factorial(E_k)/( Z_k * pow(comb(N, k), E_k) *mu ) ) * pow(((N - k + 1)/k) + (1/mu), -E_k-1)
      foldproduct *= P_H_k

      #foldproduct *= k
      print("k is %3d P_H_k %1.6f and the rolling product is %2.10f " % (k, P_H_k, foldproduct))

P_H = foldproduct
#
top = P_G_H * P_H
P_G_arr.append(top)

# run the algorithm

FILENAME = "../windsurfers.gt"
G_orig  = init( FILENAME )
if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
h_graph = init_hypergraph(G_orig)
stop_execution = input("Algorithm finished. Press any key to exit...")
# Visualisation
