# necessary packages
from           math import *
from graph_tool.all import *
from      itertools import combinations

# -> how to iterate a python set?
#   <- convert to list

### Find best hypergraph ########################

def init( filename):
    try:
        _g = load_graph( filename)
        return _g
    except:
        print("ERROR while loading", filename)
        return None

def init_hypergraph( G_orig):
    # Returns datatype: dict of frozensets of uints (mappable to graph_tool.Vertices), maps to Z+
    _hypergraph = dict()
    ### Current methodology: Adds all maximal cliques as hyperedges to the hypergraph.
    # Adds every maximal clique in G_orig as hyperedges to h_graph
    _m_cliques = max_cliques( G_orig )
    for clique in _m_cliques:
        fs = frozenset( clique )
        _hypergraph[ fs ] = 1 # + _h_graph.get(fs, 0)
    ### Alternative methods include random init, edge init, or empty (page 6, last paragraph before section [D] )
    #
    return _hypergraph

# projection component -- equation (2)

# helper function, projects a single hyperedge to edges, to the provided adjacency "matrix" i.e. dictionary
def project_hyperedge( hyperedge, adj ):
    assert type( hyperedge) == frozenset, "_hyperedge is not a frozenset"
    _gen = combinations( hyperedge, 2)
    for tup in _gen:
        _fs = frozenset(tup)
        adj[ _fs] = 1
    return

# helper function, finds the adjacency matrix (dictionary) of a graph_tool.Graph
def get_adjacency_from_Graph( G_orig):
    assert type(G_orig) == graph_tool.Graph, "G_orig is not a Graph object"
    _adj_G_orig = dict()
    for edge in G_orig.edges():
        _src = G_orig.vertex_index[edge.source()]
        _trg = G_orig.vertex_index[edge.target()]
        _adj_G_orig[ frozenset( [ _src, _trg] ) ] = 1
    return _adj_G_orig

# helper function, finds the adjacency matrix (dictionary) of a hypergraph projected down
def get_adjacency_from_hypergraph( hypergraph):
    assert type(hypergraph) == dict, "hypergraph is not a dictionary type"
    _adj_G_projected = dict()
    for hyperedge in hypergraph:
        project_hyperedge(hyperedge, _adj_G_projected)
    return _adj_G_projected

# Checks whether H projects to G, (!) note that you need to get both adjacency matrices of G and H to input into this
def get_Prob_G_if_H( adj_G_projected, adj_G_orig):
    # take a hypergraph and map all of its hyperedges into pairwise interactions and store them in adj_G_projected
    # take the original graph's edges, turn that into adj_G_orig and compare the two adjacency "matrices"
    if adj_G_orig == adj_G_projected:
        return 1
    else:
        return 0

# hypergraph prior -- equation (12)
def get_Prob_H( hypergraph, max_hyperedge_size, graph_order ):
    # todo
    _E = len(get_adjacency_from_hypergraph(hypergraph))
    _L = max_hyperedge_size
    _N = graph_order # placeholder # order of original graph? size: number of edges
    _running_product = 1

    for k in range(2, _L + 1):
        mu = _E / ( _L - 1)
        E_k  = 1 # placeholder
        Z_k  = 1 # placeholder, todo
        # subrountine here, equation (7)
        # loop through hypergraph, collect all hyperedges of size k, find eta^(k)_m
        he_k   = dict()
        l_he_k = list(he_k)
        for h_edge in l_he_k:
            maxm = 1 # placeholder

        # and calculate product
        Z_k_fold = 1
        for m in range(1, maxm + 1):
            # shenanigans
            for he in he_k:
                Z_k_fold *= pow( factorial(m), squiggle)


    #mu   = (L - 1)*log(1/(1-E/comb(N,2
    #nu_k * comb(N, k)
    #print("k is", k, mu)
    P_H_k  = ( factorial(E_k)/( Z_k * pow(comb( _N, k), E_k) *mu ) ) * pow((( _N - k + 1)/k) + (1/mu), -E_k-1)
    _running_product *= P_H_k

    # _running_product *= k
    print("k is %3d P_H_k %1.6f and the running product is %2.10f " % (k, P_H_k, _running_product))

    return _running_product # placeholder
