
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

# def print_list_magnitudes(list, end = None):
#     if end != None:
#         assert type( end) == str
#     for element in list:
#
#         print(f"{element}, ", end = '')
#     print("", end = end)

### Find best hypergraph ########################
class Hypergraph_Reconstructor:

    def __init__(self, filename, print_period = None):

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
        self._epsilon                = pow(10, -6) # cutoff point for miniscule prob calculations

        self._current_hypergraph     = dict()
        self._best_hypergraph        = self._current_hypergraph
        self._best_hyperprior        = 1
        self._P_G_arr                = []
        self._P_G                    = 1

        self._print_period           = 5 # used to control printing to console only periodically
        self._periodic_print         = self._print_period
        self._iteration              = 0

    # Sets (or resets) the current hypergraph using some initialisation method on the original graph
    # Returns void
    def init_hypergraph(self):

        self._iteration          = 0
        self._current_hypergraph = dict() # datatype: dict of frozensets of uints (mappable to graph_tool.Vertices), maps to Z+
        ### Current methodology: Adds all maximal cliques as hyperedges to the hypergraph.
        _m_cliques      = max_cliques( self._g ) # returns: iterator over numpy.ndarray
        for clique in _m_cliques:
            fs = frozenset( clique )
            self._current_hypergraph[ fs ] = 1 # + _h_graph.get(fs, 0)

            # simultaneously find out maximal hyperedge size _L
            _edge_size = len(clique)
            if _edge_size > self._maximal_hyperedge_size:
                self._maximal_hyperedge_size = _edge_size

        ### Alternative methods include random init, edge init, or empty (page 6, last paragraph before section [D] )
        #

        return

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

    ## hypergraph prior -- equation (12)
    ## returns a tuple [generator of a real number, list E_k, list Z_k ], k from 0 until maximal hyperedge size
    def get_Prob_H(self, hypergraph):

        _E_total = self._graph_edges_total
        _L       = self._maximal_hyperedge_size
        _N       = self._graph_order

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
        # turn into quickly accessible array
        _E = [ _Ed.get( i,0) for i in range(0, _L+1)]   # again, note the indexing that includes 2 dummy entries [0] and [1]
        _Z = [ _Zd.get( i,1) for i in range(0, _L+1)]
        # -----------------------

        new_mu = _E_total / ( _L - 1)
        if self._mu < new_mu:
            self._mu = new_mu
        # and calculate product
        #print("-- _mu: %d = %d / ( %d - 1) -----\n" % ( _mu, _E_total, _L ) )

        ## helper
        ## this returns values increasingly miniscule with increasing graph order, so calculating it is fairly pointless
        def get_equation_12():

            _running_product = 1
            ### realisation:
            ### you don't really need absolute probabilities, and in this situation it is untenable to calculate those miniscule probabilities. You can, instead, calculate their relative probabilities during MCMC
            def get_diminishing_product(k):
                # factorial( _E[k]) / pow(comb( _N, k), _E[k])
                bottom_  = comb( _N, k)
                E_i_     = _E[k]
                product_ = 1
                while E_i_ > 0:
                    product_ *= E_i_ / bottom_
                    if product_ < self._epsilon:
                        return [False, 0]
                    E_i_ -= 1
                return [ True, product_ ]

            for k in range(2, _L + 1):
                P_H_k  = 1
                term1  = get_diminishing_product(k)
                if term1[0]:
                    term2  = pow( 1/ self._mu + 1, - _E[k]-1) / ( _Z[k] * self._mu)
                    P_H_k  = term1[1] * term2
                else:
                    P_H_k  = self._epsilon
                _running_product *= P_H_k
                #print("E_%d: %d , %d ! = %d, Z_%3d: %4d , (%d choose %d): %d " % (k, _E[k], _E[k], factorial(_E[k]), k, _Z[k], _N, k, comb( _N, k)) )
                #print("----- k: %d P_H_k %f and the running product: %f -----\n" % (k, P_H_k, _running_product))

            yield _running_product

        return [ get_equation_12(), _E, _Z ]

    ##
    def calculate_Prob_G(self):
        sum = 0
        for ( hypergraph, hyperprior) in self._P_G_arr:
            sum += hyperprior
        self._P_G = sum

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
            raise Exception("Error: hypergraph size 0... Have you initialized the hypergraph?")
            #return (False, hypergraph, self.get_Prob_H( hypergraph)[0] )

        ## Find maximal hyperedge
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
        l = 1
        if k > 1:
            l = random.randint( 2, k)
        _sub_hyperedge = _largest
        if l < k:
            _set   =  set( _sub_hyperedge)
            _l_set = list( _sub_hyperedge)
            while len( _set) > l:
                _set.discard( _l_set[ random.randint( 0, _size_largest -1)] )
            _sub_hyperedge = frozenset( _set)

        ## ... and change the frequency of _sub_hyperedge. If its count is 0, increase by 1, else increase or decrease by 1 with 50% prob each
        #print(f"sub { _sub_hyperedge} : { _new_hypergraph.get( _sub_hyperedge, 0)}", end = '')
        if _new_hypergraph.get( _sub_hyperedge, 0) < 1:
            _new_hypergraph[ _sub_hyperedge] = 1
        else:
            if random.random() < 0.5:
                _new_hypergraph[ _sub_hyperedge] += 1
            else:
                _new_hypergraph[ _sub_hyperedge] -= 1
        # if it's now 0, clean up
        if _new_hypergraph[ _sub_hyperedge] < 1:
            #print(f"new len {len( _new_hypergraph)} -> ", end = '')
            _new_hypergraph.pop( _sub_hyperedge)
            #print(f"{len( _new_hypergraph)}")

        #print(f" -> { _new_hypergraph.get( _sub_hyperedge, 0)}")

        ## calculate the hyperpriors P(H) and compare
        ( _P_H_current, _E_current, _Z_current) = self.get_Prob_H(      hypergraph)
        ( _P_H_new    , _E_new    , _Z_new    ) = self.get_Prob_H( _new_hypergraph)

        acceptance_rate = 1
        for k in range(2, _L+1):
            term1 = factorial_div_factorial( _E_new[k], _E_current[k])
            term2 = _Z_current[k] / _Z_new[k]
            term3 = pow( (comb( _N, k) * (1/ self._mu) +1), _E_current[k] - _E_new[k])
            prod  = term1 * term2 * term3
            acceptance_rate *= prod
            #print(f"{prod} ", end = '') # debug

        _cointoss  = random.random() < acceptance_rate

        ## check whether P(G|H) = 1 holds
        _projects_to_graph = self.get_Prob_G_if_H( self.get_adjacency_from_hypergraph( _new_hypergraph))

        ## check acceptance. If heads, record the change, otherwise keep the previous hypergraph
        if _projects_to_graph and _cointoss:
            if self._periodic_print == 0:
                print(f"\n\nhyperedge { _sub_hyperedge} : { _new_hypergraph.get( _sub_hyperedge, 0)}  ", end = '')
                print(f"acceptance rate { acceptance_rate}, ", end='') #<- coin toss:{ _cointoss}")
                print(f"len { len( _new_hypergraph)}")
                print(f"Old E:{ _E_current}") # Z:{[ '{:.1E}'.format(val) for val in _Z_current ]}")
                print(f"New E:{ _E_new    }") # Z:{[ '{:.1E}'.format(val) for val in _Z_new     ]}")
                print()
                self._periodic_print = self._print_period
            self._periodic_print -= 1
            return (True, _new_hypergraph, _P_H_new)
        else:
            return (False, hypergraph, _P_H_current)


    ##
    ## main algorithm version 1, always resets progress, <- change to store currently best solution in object state
    ## returns void
    def find_best_hypergraph(self, max_iterations = None):

        #_hypergraph         = self.init_hypergraph( self._g)
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
        self._P_G     = 0

        i = 0
        while i < _MAX_ITERATONS:
            out = self.find_candidate_hypergraph( _best_hypergraph)
            if out[0]:
                print(f"Iteration {i}   ", end = '')
                _best_hypergraph      = out[1]
                gen                   = out[2]
                for val in gen: pass
                _best_hyperprior      = val
                self._best_hypergraph = _best_hypergraph
                self._best_hyperprior = val
                self._P_G_arr.append( ( _best_hypergraph, _best_hyperprior))
                self._P_G            += val
                i += 1

        return
    ## main algorithm version 2
    # change init_hypergraph(): set self._current_hypergraph to use initialisation algorithm on the original graph, returns void
    # use this method to run the algorithm for a set amount of iterations
    def run_algorithm(self, iterations = None):

        _result          = self.get_Prob_H( self._current_hypergraph)
        for val in _result[0]: pass
        _best_hyperprior = val
        _ITERATIONS      = 20
        if not iterations == None:
            assert type( iterations) == int and iterations > 0, "find_best_hypergraph() error: iterations input"
            _ITERATIONS   = iterations

        ### evidence, normalization
        # add each iteration of P_G_H * P_H to this array, and then sum them up at the end
        #self._P_G_arr = [ ( _best_hypergraph, _best_hyperprior) ]
        #self._P_G     = 0

        i = 0
        while i < _ITERATIONS:
            out = self.find_candidate_hypergraph( self._current_hypergraph)
            if out[0]:
                print(f"Iteration {self._iteration}   ", end = '')
                _best_hypergraph      = out[1]
                gen                   = out[2]
                for val in gen: pass
                _best_hyperprior         = val
                self._current_hypergraph = _best_hypergraph
                self._best_hypergraph    = self._current_hypergraph
                self._best_hyperprior    = val
                #self._P_G_arr.append( ( _best_hypergraph, _best_hyperprior))
                self._P_G            += val
                i               += 1
                self._iteration += 1

        return

    ## Observation: as of this 28.11.2022 implementation, the algorithm tends to endlessly add 2-edges. This method cuts down on the number of repeated hyperedges to a given maximum number
    ## returns a number (int) of all those pruned away

    ## edit 1: or no, it doesn't now. What is going on? Now it's stabilizing around some values
    #  edit 2: Think I've found an odd bug. After loading up the file into the object you'd normally need to use the method init_hypergraph() to set the initial state for the algorithm to run on. If you do this, the test graph windsurfers.gt will stabilize to some hypergraph of 800 2-edges. If you don't initialize and try to run the algorithm, it'll throw an error as per planned; but if you then initialize it and run the recovery algorithm, it will "stabilize" around 400 2-edges 430 3-edges. The 3-edges part is stable but the 2-edges part will endlessly rise in count.

    def prune_hypergraph(self, threshold):

        assert type( threshold)  == int and threshold >= 0

        stats = 0
        for hyperedge in self._current_hypergraph:
            count = self._current_hypergraph[ hyperedge]
            if count > threshold:
                stats += count - threshold
                self._current_hypergraph[ hyperedge] = threshold
        print(f"{stats} hyperedges pruned")
        return stats
    #
