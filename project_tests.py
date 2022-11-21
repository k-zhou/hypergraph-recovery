
from project_methods import *
# import string

def test_types_1( hypergraph):

    global l_hypergraph
    l_hypergraph = list( hypergraph)
    print("Checks types from the hypergraph")
    row = l_hypergraph[0]
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

def test_adj_equal( hypergraph, G_orig):

    print("Gets adjacency lists for the hypergraph and G_orig")
    _adj_G_orig      = get_adjacency_from_Graph( G_orig)
    _adj_G_projected = get_adjacency_from_hypergraph( hypergraph)
    output = get_Prob_G_if_H( _adj_G_projected, _adj_G_orig)
    print("Adjacent?", output)
    return

def test_types_2():

    gen = G_orig.vertex(0).out_neighbours()
    for i in gen:
        print(f"{i} {type(i)} {int(i)} {type(int(i))}")

def test_small_G( hyperedge):
    _small_G = Graph()
    global _mapping, _inv_map
    _mapping = dict() # maps from small graph index to original graph index
    _inv_map = dict() # maps from original graph index to small graph index
    _largest = hyperedge
    _l_largest = list( _largest)
    for v in range( len( _largest)):
        _mapping[ v] = _l_largest[ v]
        _inv_map[ _l_largest[ v]] = v
        _small_G.add_vertex()
    # connect the small graph according to the original graph
    for v in range( len( _largest)):
        for nv in G_orig.vertex( _mapping[ v]).out_neighbours():
            # note: nv type is Vertex. Convert to int to use as index
            _target = _inv_map.get(int(nv), -1)
            if _target != -1:
                _small_G.add_edge( v, _target)

    return ( _small_G, _mapping, _inv_map )

def testrnd1( n=None):

    _n = input("Random test: How many times? ")
    _n_default = 10000
    if not _n.isnumeric():
        # note: an empty string counts as False
        _n = _n_default
    else:
        _n = int( _n)
    if _n == 0:
        _n = _n_default
    _sum = 0
    print(f"Running for { _n} rounds.")
    for i in range( _n):
        _sum += random.random()
    print(f"The average is { _sum/ _n}")

def test_subset_manipulation():

    _startingsize = 30
    _sizemax      = 25
    _sizemin      = 15
    _targetsize   = random.randint( _sizemin, _sizemax)
    _valuemax  = 150
    _valuemin  = 1
    print(f" Generating a set with { _startingsize} starting elements between [{ _valuemin} and { _valuemax}]")
    _set = set({ random.randint( _valuemin, _valuemax) })
    while len( _set) < _startingsize:
        _set.add( random.randint( _valuemin, _valuemax))
    print( _set)
    print(f" removing elements until the set is of size { _targetsize}... ")
    _l_set = list( _set)
    while len( _set) > _targetsize:
        _set.discard( _l_set[random.randint( 0, len( _l_set) -1 )] )
    print( _set)

def run_algorithm( max_iterations = None):

    reconstructor.find_best_hypergraph( max_iterations)

### Main ##########
print("--Tests script--")
FILENAME = "../windsurfers.gt"
#G_orig   = init( FILENAME )
#if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
reconstructor = Hypergraph_Reconstructor( FILENAME)
print("Loaded file", FILENAME, "\nrun tests")
#hypergraph   = init_hypergraph(G_orig)
#graph_order  = len(list(G_orig.vertex_index))
#results      = None
