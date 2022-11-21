
from           math import comb, pow, factorial
from graph_tool.all import *
from      itertools import combinations
import       random
import        numpy as     np

# -> how to iterate through a python set?
#   <- convert to list
# graph  size: number of edges
# graph order: number of vertices

# helper: calculate  a! / b!
def factorial_div_factorial(top, bot):
    assert type(top) == int
    assert type(bot) == int
    assert top >= 0
    assert bot >= 0
    if top > bot:
        i   = top
        ans = 1
        while ( i > bot ):
            ans *= i
            i   -= 1
        return ans
    elif top == bot:
        return 1
    else:
        return 1/factorial_div_factorial(bot, top)

### Find best hypergraph ########################
class Hypergraph_Reconstructor:

    def __init__(self, filename):

        self._filename = filename
        try:
            self._g = load_graph( self._filename) # graph_tool.Graph
        except:
            raise Exception("ERROR while loading " + filename)

        self._adj_graph              = self.get_adjacency_from_Graph(self._g) # adjacency list as a dict( frozensets) -> Z+
        self._graph_edges_total      = len(self._adj_graph) # the total number of edges in the graph
        self._graph_order            = len(list(self._g.vertex_index))
        self._maximal_hyperedge_size = 2  # to be determined later while running the algorithm
        self._mu                     = 1  # equation (11), to be determined later

        self._best_hypergraph        = dict()
        self._best_hyperprior        = 1
        self._P_G_arr                = []

    # Returns datatype: dict of frozensets of uints (mappable to graph_tool.Vertices), maps to Z+
    def init_hypergraph(self, graph):

        _hypergraph = dict()
        ### Current methodology: Adds all maximal cliques as hyperedges to the hypergraph.
        _m_cliques = max_cliques( graph ) # returns: iterator over iterator over numpy.ndarray
        for clique in _m_cliques:
            fs = frozenset( clique )
            _hypergraph[ fs ] = 1 # + _h_graph.get(fs, 0)
        ### Alternative methods include random init, edge init, or empty (page 6, last paragraph before section [D] )
        #
        return _hypergraph

    ## projection component -- equation (2)

    ## helper function, projects a single hyperedge to edges, to the provided adjacency list
    ## returns void
    def project_hyperedge(self, hyperedge, adj ):

        assert type( hyperedge) == frozenset, "_hyperedge is not a frozenset"
        _gen = combinations( hyperedge, 2)
        for tup in _gen:
            _fs = frozenset(tup)
            adj[ _fs] = 1
        return

    # helper function, finds the adjacency list (dictionary) of a graph_tool.Graph
    # returns a dict( frozenset() ) -> int Z+
    def get_adjacency_from_Graph(self, graph):

        assert type(graph) == graph_tool.Graph, "the input is not a Graph object"
        _adj_graph = dict()
        for edge in graph.edges():
            _src = graph.vertex_index[edge.source()]
            _trg = graph.vertex_index[edge.target()]
            _adj_graph[ frozenset( [ _src, _trg] ) ] = 1
        return _adj_graph

    # helper function, finds the adjacency list (dictionary) of a hypergraph projected down
    # returns a dict( frozenset() ) -> int Z+
    def get_adjacency_from_hypergraph(self, hypergraph):

        assert type(hypergraph) == dict, "hypergraph is not a dictionary type"
        _adj_G_projected = dict()
        for hyperedge in hypergraph:
            if hypergraph[ hyperedge] > 0:
                self.project_hyperedge(hyperedge, _adj_G_projected)
        return _adj_G_projected

    # Checks whether H projects to G, (!) note that you need to get both adjacency matrices of G and H to input into this
    # returns 0 or 1
    def get_Prob_G_if_H(self, adj_G_projected):
        # take a hypergraph and map all of its hyperedges into pairwise interactions and store them in adj_G_projected
        # take the original graph's edges, turn that into adj_G_orig and compare the two adjacency lists
        if self._adj_graph == adj_G_projected:
            return 1
        else:
            return 0

    # hypergraph prior -- equation (12)
    # returns a tuple [real number, list E_k, list Z_k ], k from 0 until maximal hyperedge size
    def get_Prob_H(self, hypergraph):

        _E_total = self._graph_edges_total
        _L       = self._maximal_hyperedge_size
        _N       = self._graph_order
        _running_product = 1

        # subrountine here ------
        # -- equation (5)
        _Ed = dict()
        # -- equation (7)
        _Zd = dict()
        # iterate through all hyperedges and find
        # -> Z_k, here collectively in an array _Z[] length _L +1, note the indexing: _Z[i] gives Z_i
        # -> E_k, -ii-
        for hyperedge in hypergraph:
            # (5)
            _edge_size = len(hyperedge)
            _Ed[ _edge_size] = _Ed.get( _edge_size, 0) + hypergraph[ hyperedge]
            # (7)
            _Zd[ _edge_size] = _Zd.get( _edge_size, 1) * factorial( hypergraph[ hyperedge])
            # simultaneously find out maximal hyperedge size _L
            if _edge_size > _L:
                _L = _edge_size
        # turn into quickly accessible array
        _E = [ _Ed.get( i,0) for i in range(0, _L+1)]   # again, note the indexing that includes 2 dummy entries [0] and [1]
        _Z = [ _Zd.get( i,1) for i in range(0, _L+1)]
        # -----------------------

        new_mu = _E_total / ( _L - 1)
        if self._mu < new_mu:
            self._mu = new_mu
        # and calculate product
        #print("-- _mu: %d = %d / ( %d - 1) -----\n" % ( _mu, _E_total, _L ) )
        for k in range(2, _L + 1):
            P_H_k  = ( factorial( _E[k])/( _Z[k] * self._mu * pow(comb( _N, k), _E[k]) ) ) * pow( 1/ self._mu + 1, - _E[k]-1)
            _running_product *= P_H_k
            #print("E_%d: %d , %d ! = %d, Z_%3d: %4d , (%d choose %d): %d " % (k, _E[k], _E[k], factorial(_E[k]), k, _Z[k], _N, k, comb( _N, k)) )
            #print("----- k: %d P_H_k %f and the running product: %f -----\n" % (k, P_H_k, _running_product))

        return [ _running_product, _E, _Z ]

    ## plan: to find a clique within a given hyperedge you need to create a new smaller Graph and a mapping which takes the indices of the vertices of the original Graph and maps them to those of the smaller Graph. You also simultaneously need an inverse map
    ## this way you can use the Graph.max_cliques() method on the small graph

    ## Markov Chain Monte Carlo: Metropolis-Hastings algorithm, find new hypergraph candidates, section (C)
    ## returns a tuple (Bool, dict( frozenset() ) -> int Z+, float )
    ## where
     # Bool is True when the hypergraph is different from the input
     #
     # float is the hyperprior of the hypergraph
    def find_candidate_hypergraph(self, hypergraph ):

        _N  = self._graph_order
        _L  = self._maximal_hyperedge_size

        if len( hypergraph) == 0:
            print("WARNING: hypergraph size 0")
            return (False, hypergraph, self.get_Prob_H( hypergraph)[0] )

        ## Find maximal hyperedge
        _EPSILON        = 0.000001 # used for checking tiny probabilities of float types
        _new_hypergraph = hypergraph.copy()
        _largest        = frozenset({})
        _size_largest   = len( _largest)
        for hyperedge in _new_hypergraph:
            if len( hyperedge) > _size_largest:
                _largest      = hyperedge
                _size_largest = len( _largest)

        ## Find the size of its largest clique
        ## generate a smaller graph and operate on that
        _small_G = Graph()
        _mapping = dict() # maps from small graph index to original graph index
        _inv_map = dict() # maps from original graph index to small graph index
        _l_largest = list( _largest)
        for v in range( _size_largest):
            _mapping[ v] = _l_largest[ v]
            _inv_map[ _l_largest[ v]] = v
            _small_G.add_vertex()
        ## connect the small graph according to the original graph
        for v in range( len( _largest)):
            for nv in self._g.vertex( _mapping[ v]).out_neighbours():
                # note: nv type is Vertex. Convert to int to use as index
                _target = _inv_map.get(int(nv), -1)
                if _target != -1:
                    _small_G.add_edge( v, _target)
        ## find the size of the largest clique in this smaller graph
        _mqs = max_cliques( _small_G)
        k = 0
        for mq in _mqs:
            if len(mq) > k:
                k = len(mq)

        ## select a random sub-hyperedge within this hyperedge (can select itself) ...
        l = random.randint( 1, k)
        _sub_hyperedge = _largest
        if l < k:
            _set   =  set( _sub_hyperedge)
            _l_set = list( _sub_hyperedge)
            while len( _set) > l:
                _set.discard( _l_set[ random.randint( 0, _size_largest -1)] )
            _sub_hyperedge = frozenset( _set)

        ## ... and change the frequency of _sub_hyperedge. If its count is 0, increase by 1, else increase or decrease by 1 with 50% prob each
        print(f"sub { _sub_hyperedge} : { _new_hypergraph.get( _sub_hyperedge, 0)}", end = '')
        if _new_hypergraph.get( _sub_hyperedge, 0) < 1:
            _new_hypergraph[ _sub_hyperedge] = 1
        else:
            if 0.5 < random.random():
                _new_hypergraph[ _sub_hyperedge] += 1
            else:
                _new_hypergraph[ _sub_hyperedge] -= 1
        # if it's now 0, clean up
        if _new_hypergraph[ _sub_hyperedge] < 1:
            _new_hypergraph.pop( _sub_hyperedge)

        print(f" -> { _new_hypergraph.get( _sub_hyperedge, 0)}")

        ## calculate the hyperpriors P(H) and compare
        ( _P_H_current, _E_current, _Z_current) = self.get_Prob_H(      hypergraph)
        ( _P_H_new    , _E_new    , _Z_new    ) = self.get_Prob_H( _new_hypergraph)
        print(f"New E:{ _E_new} Z:{ _Z_new}")
        _ratio       = 0
        _cointoss    = False

        ## check for tiny probabilities
        ##### TODO
        # FIGURE THIS OUT NEXT

        acceptance_rate = 1
        for k in range(2, _L+1):
            term1 = factorial_div_factorial( _E_new[k], _E_current[k])
            term2 = _Z_current[k] / _Z_new[k]
            term3 = pow( (comb( _N, k) * (1/ self._mu) +1), _E_current[k] - _E_new[k])
            acceptance_rate *= term1 * term2 * term3
        print(f"acceptance rate { acceptance_rate} ", end = '')

        if _P_H_current < _EPSILON:
            _ratio     = 1
            _cointoss  = True
        else:
            _ratio     = _P_H_new / _P_H_current
            _cointoss  = _ratio < random.random()

        ## check whether P(G|H) = 1 holds
        _projects_to_graph = self.get_Prob_G_if_H( self.get_adjacency_from_hypergraph( _new_hypergraph))

        ## check acceptance. If heads, record the change, otherwise keep the previous hypergraph
        #print(f"projects: { _projects_to_graph}, coin toss:{ _cointoss}")
        if _projects_to_graph and _cointoss:
            return (True, _new_hypergraph, _P_H_new)
        else:
            return (False, hypergraph, _P_H_current)

    ## main algorithm
    # WIP
    ## returns void
    def find_best_hypergraph(self, max_iterations = None):

        _hypergraph         = self.init_hypergraph( self._g)
        _best_hypergraph    = _hypergraph
        _result             = self.get_Prob_H( _hypergraph)
        _best_hyperprior    = _result[0]
        _MAX_ITERATONS      = 0
        if not max_iterations == None:
            assert type(max_iterations) == int and max_iterations > 0, "find_best_hypergraph() error: max_iterations input"
            _MAX_ITERATONS  = max_iterations
        else:
            _MAX_ITERATONS  = 50

        ### evidence, normalization
        # add each iteration of P_G_H * P_H to this array, and then sum them up at the end
        self._P_G_arr = [ ( _best_hypergraph, _best_hyperprior) ]

        i = 0
        while i < _MAX_ITERATONS:
            print(f"Iteration {i}")
            out = self.find_candidate_hypergraph( _best_hypergraph)
            if out[0]:
                self._best_hypergraph = out[1]
                self._best_hyperprior = out[2]
                self._P_G_arr.append( ( _best_hypergraph, _best_hyperprior))
            i += 1

        return
    #
