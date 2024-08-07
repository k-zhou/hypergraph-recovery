
from           math import comb, pow, factorial, ceil
from     statistics import stdev
from graph_tool.all import *
from      itertools import combinations
from           time import time_ns, sleep
import       random
#import        numpy     as np

from helper_functions import *

# graph  size: number of edges
# graph order: number of vertices
# How to program an automatic detection for reaching an equilibrium that only fluctuates little?

############### Helpers #################

################ Main ################
class Hypergraph_Reconstructor:

    def init_hypergraph (self               ) -> None: pass
    def add_to_history  (self, str_data = "") -> None: pass
    def add_to_history_exact(self, data = frozenset()) -> None: pass
    def add_to_log      (self, str_data = "") -> None: pass

    def __init__(self, filename, print_period = 1):

        self._filename = filename
        try:
            self._g = load_graph( self._filename) # graph_tool.Graph
        except:
            raise Exception("ERROR while loading " + filename)
        self._filename_pathless      = strip_filename_path(self._filename)
        self._file_path              = strip_filename_suffix(self._filename, '/') + '/'
        self._filename_only          = strip_filename_suffix(self._filename_pathless)

        ## Graph information
        self._adj_graph              = self.get_adjacency_from_Graph(self._g) # adjacency list as a dict( frozensets) -> Z+
        self._graph_edges_total      = self._g.num_edges()
        self._graph_order            = self._g.num_vertices()
        self._maximal_hyperedge_size = 2  # to be determined later while running the algorithm, also update _stopping_arr
        self._mu                     = 1  # equation (11), to be determined later
        self._epsilon                = pow(10, -6) # cutoff point for miniscule prob calculations

        self._current_hypergraph     = dict()
        #self._best_hypergraph        = self._current_hypergraph
        self._best_hyperprior        = 1
        self._P_G_arr                = [] # mostly unused, holds tuples of (hypergraph, hyperprior (Int))
        self._P_G                    = 1
        self._P_H_current            = 0
        
        self._Z_current              = []

        ## the current hypergraph's tally of hyperedges of size k, ordered by index
        ## i.e. [2] holds the tally of 2-edges, [3] holds the tally of 3-edges etc.
        self._E_current              = []
        self._diff_E                 = (0, 0)

        ## For use in automatically stopping the algorithm.
        ## Reasoning: When this algorithm approaches an equilibrium, the average of changes in
        ## hyperedge counts, - regardless of the size of the hyperedge, will approach zero.
        ## For this, use an array as a rolling window to keep track of the changes in the tally of all sizes of hyperedges
        ## in the past 100 iterations, sum them up, and use an error margin of 5% to stop the algorithm automatically.

        ## _stopping_arr needs a more specific dimension that is the same as _E_k or _E_current, to be defined later together with _maximal_hyperedge_size
        ## (v, k) v takes the two values  [-1, +1], k takes the int values [0, k_max] where k_max is the _maximal_hyperedge_size
        ## At every iteration, from the array _rolling_window, take the tuple (v, k) pointed to by index _rw_index,
        ## subtract the value v from the array _stopping_arr at index k, replace (v, k) with the change in _E_k that leads to _E_current,
        ##  add this new value v_new to _stopping_arr at k_new given by _diff_E
        ## advance _rw_index, and wrap back around if past index bounds,
        ## calculate the folded sum of _stopping_arr and stop the algorithm when it approaches 0.
        self._stopping_arr           = []
        self._rw_size                = 100
        self._rolling_window         = [ (0, 0) for i in range(0, self._rw_size) ]
        self._rw_index               = 0
        self._stopping_sum           = 0
        self._auto_stopped           = False

        ## Iteration data
        self._iter_failed_attempts   = 0
        self._iter_failure_autostop_threshold = 10 # activates after failing this many attempts at accepting a candidate h-graph # TODO: Arbitrary
        self._iter_pass_data         = dict()
        self._last_successful_iter_time = 0

        ## Runtime-based auto-stop, used for big graphs taking longer than a few seconds per accepted iteration change
        ## this uses probability- and normal distribution -based stopping
        self._iter_runtimes          = []
        self._iter_runtimes_summed   = [0]

        ## keeping running stats
        self._print_period           = print_period # used to control printing to console only periodically, static
        self._print_clock            = time_ns() # TODO: unused
        self._iteration              = 0
        self._runtime                = 0 # runtime of the algorithm in ns
        self._hypergraph_initiated   = 0 # TODO: redundant
        self._history                = [] # used to track the changes of h-edges during each iteration, each element corresponding to
                                            # a string row when outputting to a txt file.
                                            # The first line is the initial count of different sized h-edges i.e. 
                                            # the same format as the array E: n0 n1 n2 n3 e.g.
                                            # 0 0 30 5 4 2
                                            # 0 0 295 23 10 2 1 1
                                            # Every line after that is the size of the h-edge being added/removed with a + or - e.g. :   
                                            # 2 +
                                            # 5 -
        self._history_num_arr        = [] # used for plotting, each element at index n corresponds to the array E at iteration n
        self._history_exact          = [] # keeps track of hyperedge changes. Its format is tuples (hyperedge as frozenset, new count)

        self._log                    = [] # a list of types convertible to string, later exportable to a log file

        self._log.append(self._filename_pathless)
        print(f"\tFile: {self._filename} :", end='')
        self.init_hypergraph()

    ##########################

    def get_log(self):
        return self._log


    ## Sets (or resets) the current hypergraph using some initialisation method on the original graph
    def init_hypergraph(self) -> None:

        _init_E           = dict() # Keeps track of the count of all h-edges sized n in the initial state, refer to self._history
        current_E_arr_wip = [] # use as linked list, collect all the hyperedges as their size here for the zeroth iteration

        ### Current methodology: Adds all maximal cliques as hyperedges to the hypergraph.
        _m_cliques        = max_cliques( self._g ) # returns: iterator over numpy.ndarray
        for clique in _m_cliques:
            fs = frozenset( clique )
            self._current_hypergraph[ fs ] = 1 # + _h_graph.get(fs, 0)

            ## simultaneously find out maximal hyperedge size _L
            _edge_size = len(clique)
            current_E_arr_wip.append(_edge_size)
            if _edge_size > self._maximal_hyperedge_size:
                self._maximal_hyperedge_size = _edge_size

            ## also fill in the first row of self._history
            _init_E[_edge_size] = _init_E.get(_edge_size, 0) + 1
        ## continues ...
        ## logs the initial state in the text file
        h_line = ""
        for i in range(self._maximal_hyperedge_size + 1):
            h_line += f"{_init_E.get(i, 0)} "
        self._history.append(h_line)
        self._history_exact.append(h_line)
        self._history_num_arr.append([_init_E.get(i, 0) for i in range(self._maximal_hyperedge_size + 1)])
        ## inits the stopping algorithm based on the moving window
        self._stopping_arr  = [ 0 for i in range(0, self._maximal_hyperedge_size + 1) ]
        self._E_current     = [ 0 for i in range(0, self._maximal_hyperedge_size + 1) ]
        for e_size in current_E_arr_wip:
            self._E_current[e_size] += 1 #

        ### Alternative methods include random init, edge init, or empty (page 6, last paragraph before section [D] )
        #
        ### find hyperprior P(H) of the current
        ( self._P_H_current, self._E_current, self._Z_current) = self.get_Prob_H( self._current_hypergraph)
        
        self._hypergraph_initiated = 1
        print("Initialised hypergraph.")

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

    ## helper function, finds the adjacency list (dictionary) of a graph_tool.Graph
    ## returns a dict( frozenset() ) -> int Z+
    def get_adjacency_from_Graph(self, graph) -> dict:

        assert type(graph) == graph_tool.Graph, "the input is not a Graph object"
        _adj_graph = dict()
        for edge in graph.edges():
            _src = graph.vertex_index[edge.source()]
            _trg = graph.vertex_index[edge.target()]
            _adj_graph[ frozenset( [ _src, _trg] ) ] = 1
        return _adj_graph

    ## helper function, finds the adjacency list (dictionary) of a hypergraph projected down
    ## returns a dict( frozenset() ) -> int Z+
    def get_adjacency_from_hypergraph(self, hypergraph) -> dict():

        assert type(hypergraph) == dict, "hypergraph is not a dictionary type"
        _adj_G_projected = dict()
        for hyperedge in hypergraph:
            if hypergraph[ hyperedge] > 0:
                self.project_hyperedge(hyperedge, _adj_G_projected)
        return _adj_G_projected

    ## Checks whether H projects to G, (!) note that you need to get both adjacency matrices of G and H to input into this
    ## returns 0 or 1
    def get_Prob_G_if_H(self, adj_G_projected):
        ## take a hypergraph and map all of its hyperedges into pairwise interactions and store them in adj_G_projected
        ## take the original graph's edges, turn that into adj_G_orig and compare the two adjacency lists
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

        ## subrountine here ------
        ## -- equation (5)
        _Ed = dict()
        ## -- equation (7)
        _Zd = dict()
        ## iterate through all hyperedges and find
        ## -> Z_k, here collectively in an array _Z[] length _L +1, note the indexing: _Z[i] gives Z_i
        ## -> E_k, -ii-
        for hyperedge in hypergraph:
            ## (5)
            _edge_size = len(hyperedge)
            _Ed[ _edge_size] = _Ed.get( _edge_size, 0) + hypergraph[ hyperedge]
            ## (7)
            _Zd[ _edge_size] = _Zd.get( _edge_size, 1) * factorial( hypergraph[ hyperedge])
        ## turn into quickly accessible array
        _E = [ _Ed.get( i,0) for i in range(0, _L+1)]   # again, note the indexing that includes 2 dummy entries [0] and [1]
        _Z = [ _Zd.get( i,1) for i in range(0, _L+1)]
        ## -----------------------

        new_mu = _E_total / ( _L - 1)
        if self._mu < new_mu:
            self._mu = new_mu
        ## and calculate product
        #print("-- _mu: %d = %d / ( %d - 1) -----\n" % ( _mu, _E_total, _L ) )

        ## helper
        ## this returns values increasingly miniscule with increasing graph order, so calculating it is fairly pointless
        def get_equation_12():

            _running_product = 1
            ### realisation:
            ### you don't really need absolute probabilities, and in this situation it is untenable to calculate those miniscule probabilities. You can, instead, calculate their relative probabilities during MCMC
            def get_diminishing_product(k):
                ## factorial( _E[k]) / pow(comb( _N, k), _E[k])
                bottom_  = comb( _N, k)
                E_i_     = _E[k]
                product_ = 1
                while E_i_ > 0:
                    product_ *= E_i_ / bottom_
                    if product_ < self._epsilon:
                        return [False, self._epsilon]
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
    ## Bool is True when the hypergraph is different from the input
    ##
    ## float is the hyperprior of the hypergraph
    def find_candidate_hypergraph(self, index=0, pass_data=dict()):

        _N  = self._graph_order
        _L  = self._maximal_hyperedge_size

        if len(self._current_hypergraph) == 0:
            raise Exception("Error: hypergraph size 0... Have you initialized the hypergraph?")
            #return (False, hypergraph, self.get_Prob_H( hypergraph)[0] )
        
        ## variables to display status
        pass_data["start_time"]   = \
        last_print_time           = time_ns()
        print_time_fail_interval  = 15000000000 # 15 seconds
        next_print_time_fail      = last_print_time + print_time_fail_interval

        ## Find maximal hyperedge
        _new_hypergraph = self._current_hypergraph.copy()
        _largest        = frozenset()
        _size_largest   = 0
        for hyperedge in _new_hypergraph:
            if len( hyperedge) > _size_largest and _new_hypergraph[hyperedge] > 0:
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
                ## note: nv type is Vertex. Convert to int to use as index
                _target = _inv_map.get(int(nv), -1)
                if _target != -1:
                    _small_G.add_edge( v, _target)
        ## find the size of the largest clique in this smaller graph
        _mqs = max_cliques( _small_G)
        k = 0
        for mq in _mqs:
            if len(mq) > k:
                k = len(mq)

        ## loop until an accepted move is found; if 10000 attempts have been made, forcefully quit
        attempts       = 0
        while attempts < 10000:
            ## select a random sub-hyperedge within this hyperedge (can select itself) ...
            _sub_hyperedge = _largest
            l = 1
            if k > 1:
                l = random.randint( 2, k)
            if l < k:
                _set   =  set( _sub_hyperedge)
                _l_set = list( _sub_hyperedge)
                while len( _set) > l:
                    _set.discard( _l_set[ random.randint( 0, _size_largest -1)] )
                _sub_hyperedge = frozenset( _set)
            

            ## ... and change the frequency of _sub_hyperedge in the original hypergraph. 
            ## If its count is 0, increase by 1, else increase or decrease by 1 with 50% prob each

            #print(f"sub { _sub_hyperedge} : { _new_hypergraph.get( _sub_hyperedge, 0)}", end = '')
            _prev_count = _new_hypergraph.get( _sub_hyperedge, 0)
            termQ       = 1 # for checking { acceptance_rate }, pre-print eqn # 15
            if _new_hypergraph.get( _sub_hyperedge, 0) < 1:
                _new_hypergraph[ _sub_hyperedge] = 1
                termQ = 0.5
            else:
                if random.random() < 0.5:
                    _new_hypergraph[ _sub_hyperedge] += 1
                else:
                    _new_hypergraph[ _sub_hyperedge] -= 1
                if _new_hypergraph.get( _sub_hyperedge, 0) == 0:
                    termQ = 2
                else:
                    termQ = 1
            
            _count = _new_hypergraph[ _sub_hyperedge]

            ## calculate the hyperprior P(H) and compare to previous
            ( _P_H_new, _E_new, _Z_new) = self.get_Prob_H( _new_hypergraph)

            acceptance_rate = 1
            for k in range(2, _L+1):
                ## termQ  declared earlier
                term1 = factorial_div_factorial( _E_new[k], self._E_current[k])
                term2 = self._Z_current[k] / _Z_new[k]
                term3 = pow( (comb( _N, k) * (1/ self._mu) +1), self._E_current[k] - _E_new[k])
                prod  = termQ * term1 * term2 * term3
                acceptance_rate *= prod
                #print(f"{prod} ", end = '') ## debug

            _cointoss  = random.random() < acceptance_rate

            ## check whether P(G|H) = 1 holds
            _projects_to_graph = self.get_Prob_G_if_H( self.get_adjacency_from_hypergraph( _new_hypergraph))

            ## text for printing progress status
            # s1 = f"\nIteration {self._iteration}\n" + \
            #      f"hyperedge { _sub_hyperedge} : { _new_hypergraph.get( _sub_hyperedge, 0)}  " + \
            #      f"acceptance rate { acceptance_rate}, " + \
            #      f"len { len( _new_hypergraph)}\n"
            #     #acceptance rate <- coin toss:{ _cointoss}")
            #     #f"Old E:{ _E_current}") # Z:{[ '{:.1E}'.format(val) for val in _Z_current ]}"
            #     #f"New E:{ _E_new    }") # Z:{[ '{:.1E}'.format(val) for val in _Z_new     ]}"
            # s2 = f"New E:{ _E_new    } Difference:{[ (_E_new[i] - self._E_current[i]) for i in range(len(_E_new))]}\n"
            # pass_data["progress_str"] = s1 + s2

            ## check acceptance. If heads, record the change, otherwise keep the previous hypergraph
            if _projects_to_graph and _cointoss:
                
                ## if it's now 0, clean up
                if _count < 1: _new_hypergraph.pop( _sub_hyperedge)

                ## Data to pass outwards
                ## Prepares data to add to history. Refer to initialisation of self._history
                change_i = 0
                change_sign = ""
                for i in range(len(_E_new)):
                    change = _E_new[i] - self._E_current[i]
                    if change == 0:
                        continue
                    else:
                        change_i = i
                        if change == 1:
                            change_sign = "+"
                        else:
                            change_sign = "-"
                        break
                pass_data["history_str"] = str(change_i) + ' ' + change_sign
                pass_data["history_num_arr"] = _E_new.copy()
                pass_data["end_time"]   = time_ns()
                pass_data["log_str"] = f"Iteration {self._iteration} New: {str(_E_new)} diff:{str(self._diff_E)} Auto-stop state: {self._stopping_arr}; "

                ## data for auto-stop
                diff_arr            = [ _E_new[i] - self._E_current[i] for i in range(0, len( _E_new)) ]
                for i in range(0, len(diff_arr)):
                    e_size = diff_arr[i]
                    if not e_size == 0:
                        self._diff_E = (e_size, i)
                        break
                
                self._E_current = _E_new
                ( self._P_H_current, self._E_current, self._Z_current) = ( _P_H_new, _E_new, _Z_new)
                found_accepted = True
                return (True, _new_hypergraph, _P_H_new, _sub_hyperedge, _count)
            else:
                # restore the initial state and try again
                _new_hypergraph[ _sub_hyperedge] = _prev_count
                if _prev_count < 1: _new_hypergraph.pop( _sub_hyperedge)

                ## Print the status at most once every 15 seconds (and also save to output logs)
                t = time_ns()
                if t > next_print_time_fail:
                    s = f"15s elapsed. No successes after {attempts} failed attempts on iteration {self._iteration}.\n" + self._iter_pass_data.get("log_str", "No succeed iterations yet.")
                    print(s)
                    next_print_time_fail = t + print_time_fail_interval
            
            attempts += 1
        ## Check for the odd dataset that never gets iterated beyond initialisation
        return (False, self._current_hypergraph, self._P_H_current, frozenset(), 0)

    ## main algorithm version 2
    ## use this method to run the algorithm for a set amount of iterations
    def run_algorithm(self, iterations = 100, autostop = True, min_iterations = None):

        ## needs init_hypergraph() to be run first, otherwise will create unexpected behaviour
        if self._hypergraph_initiated != 1:
            #print(f"Init hypergraph: {self._filename_only}")
            self.init_hypergraph()
            self.status()

        _result          = self.get_Prob_H( self._current_hypergraph)
        for val in _result[0]: pass
        _best_hyperprior = val
        assert type( iterations) == int and iterations > 0, "run_algorithm() error: iterations input"
        _ITERATIONS   = iterations
        _autostop_enabled = bool(autostop)
        _min_iterations   = self._rw_size*2
        if not min_iterations == None:
            _min_iterations = int(min_iterations)

        ### evidence, normalization, - unused due to infinitesimal values
        ## add each iteration of P_G_H * P_H to this array, and then sum them up at the end
        #self._P_G_arr = [ ( _best_hypergraph, _best_hyperprior) ]
        #self._P_G     = 0

        def autostop(additional_text = ""):
            self._auto_stopped = True
            s                  = f"Auto-stopped algorithm at {self._iteration}th iteration.\n"
            print(s + additional_text)
            self._log.append(s + additional_text + '\n')
        
        ### run for a set amount of iterations
        
        i               = 0
        self._iter_failed_attempts = 0
        start_time      = \
        self._last_successful_iter_time = \
        last_print_time = time_ns()
        # print_time_fail_interval = 15000000000 # 15 seconds
        # next_print_time_fail = last_print_time + print_time_fail_interval
        while i < _ITERATIONS:
            
            out               = self.find_candidate_hypergraph(0, self._iter_pass_data)
            
            if out[0]:

                t                 = self._iter_pass_data["end_time"]
                iteration_runtime = t - self._last_successful_iter_time
                self._iter_runtimes.append(iteration_runtime)
                self._iter_runtimes_summed.append(iteration_runtime + self._iter_runtimes_summed[-1])

                ## Logging
                s = self._iter_pass_data["history_str"]
                self.add_to_history(s)
                self._history_num_arr.append(self._iter_pass_data["history_num_arr"])
                sub_hyperedge = out[3]
                count         = out[4]
                self.add_to_history_exact((list(sub_hyperedge), count))

                ## Print the status at most once every 3 seconds and also save to output logs
                if t - last_print_time > 3000000000:
                    s = f"{iteration_runtime // 1000000} ms elapsed until this iteration.\n" + self._iter_pass_data["log_str"]
                    print(s)
                    self.add_to_log(s)
                    last_print_time = t
                
                self._last_successful_iter_time  = t
                self._iter_failed_attempts       = 0
                _current_hypergraph      = out[1]
                gen                      = out[2]
                for val in gen: pass
                _best_hyperprior         = val
                self._current_hypergraph = _current_hypergraph
                #self._best_hypergraph    = self._current_hypergraph
                self._best_hyperprior    = val
                #self._P_G_arr.append( ( _best_hypergraph, _best_hyperprior))
                self._P_G               += val
                
                i                       += 1
                self._iteration         += 1

                ## Auto-stopping
                tup  = self._rolling_window[self._rw_index]
                self._stopping_arr[tup[1]] -= tup[0]

                tup  = self._diff_E
                self._rolling_window[self._rw_index] = tup
                self._stopping_arr[tup[1]] += tup[0]

                self._rw_index          += 1
                if self._rw_index >= self._rw_size:
                    self._rw_index = 0

                ## 2024.06.27 Observation: 
                ## A change in the tally of k-sized hyperedges can be matched by an 
                ## equivalent but negative change in the tally of l-sized hyperedges.
                ## Fold-summing these changes together might activate the 
                ## auto-stop scheme even though a local optimum has not yet been reached.
                ## For this reason a change is implemented, now it checks all sizes to 
                ## lie within the bounds
                ## - despite even this is also only an approximation to detecting reaching local optima.
                
                within_bounds = True
                for tally in self._stopping_arr:
                    if not (-3 < tally and tally < 3):
                        within_bounds = False
                        break
                
                ## auto-stop the algorithm
                if _autostop_enabled and i >= _min_iterations and within_bounds:
                    autostop()
                    break

                ## auto-stop based on runtime
                if self._iteration > 10:
                    mean                     = self._iter_runtimes_summed[self._iteration] / self._iteration
                    sd                       = stdev(self._iter_runtimes, mean)
                    if iteration_runtime > mean + 2*sd:
                        self._iter_failed_attempts  += 1
                        if self._iter_failed_attempts > self._iter_failure_autostop_threshold:
                            s = f"[!] ... due to {self._iter_failed_attempts} failed attempts after {iteration_runtime} ns with sample mean {mean} and stdeviation {sd}"
                            autostop(s)
                            print(s)
                            self.add_to_log(s)
                            break
                        else:
                            s = f"(!) {self._iter_failed_attempts} timed-out attempts by 2 * std deviation, iteration {self._iteration}"
                            print(s)
            else:
                ## Check for the odd dataset that never gets iterated beyond initialisation
                s = f"[!] 10000 failed attempts reached at iteration {self._iteration}. Auto-stopping algorithm.\n"
                autostop(s)
                break


        end_time   = time_ns()
        runtime    = end_time - start_time
        self._runtime += runtime
        self._log.append(f"More iterations took {runtime} ns or {runtime // 1000000} ms.")

        return

    ##### auxilliary methods #####
    
    ## logging

    def add_to_history(self, str_data = "") -> None:
        lines = [] # allows for passing lists of strings
        lines.append(str_data)
        for line in lines:
            self._history.append(line)

    def add_to_history_exact(self, data = frozenset()) -> None:
        lines = [] # allows for passing lists of strings
        lines.append(str(data))
        for line in lines:
            self._history_exact.append(line)

    def add_to_log(self, str_data = "") -> None:
        lines = [] # allows for passing lists of strings
        lines.append(str_data)
        for line in lines:
            self._log.append(line)

    def output_to_log(self, fname = None) -> None:
        if fname == None:
            fname = self._file_path + self._filename_only + "(log)" + ".txt"
        data_to_write = ""
        for item in self._log:
            data_to_write += str(item) + '\n'
        ## additional final details
        data_to_write += f"Total iterations {self._iteration}\n"
        data_to_write += f"Total algorithm runtime {self._runtime} ns\n  or {self._runtime / 1000000} ms"

        write_to_file(fname, data_to_write)
        return
    
    def output_history_to_log(self, fname = None) -> None:
        if fname == None:
            fname = self._file_path + self._filename_only + "(history)" + ".txt"
        data_to_write = ""
        for item in self._history:
            data_to_write += str(item) + '\n'
            
        write_to_file(fname, data_to_write)
        return
    
    def output_history_exact_to_log(self, fname = None) -> None:
        if fname == None:
            fname = self._file_path + self._filename_only + "(history_exact)" + ".txt"
        data_to_write = ""
        for item in self._history_exact:
            data_to_write += str(item) + '\n'

        write_to_file(fname, data_to_write)
        return

    def output_hypergraph_to_log(self, fname = None) -> None:
        if fname == None:
            fname = self._file_path + self._filename_only + "(h_graph)" + ".txt"
        data_to_write = ""
        data_to_write += self._filename_only + '\n'
        data_to_write += dict_fs_to_txt(self._current_hypergraph)
            
        write_to_file(fname, data_to_write)
        return
    
    ## Graph-related methods

    ## for use when the original graph loaded consists of unconnected subgraphs, this creates a new graph that finds the largest subgraph out of the original graph
    ## returns a new Graph object
    def get_largest_subgraph(self): # -> Graph
        ## this holds the order of the subgraph found when iterating a search started at a vertex i
        orders_dict = dict()
        for i in range(self._graph_order):
            if orders_dict.get(i, 0) == 0:
                ## DFS traversal
                order           = 1
                traversed_nodes = [i]
                to_be_traversed = [int(vertex) for vertex in self._g.vertex(i).out_neighbours()]
                while len(to_be_traversed) > 0:
                    node = to_be_traversed.pop()
                    if not node in traversed_nodes:
                        order += 1
                        traversed_nodes.append(node)
                        for vertex in self._g.vertex(node).out_neighbours(): to_be_traversed.append(int(vertex))
                for node in traversed_nodes:
                    orders_dict[node] = order
        ## TODO: turn this dict into a new graph object

    ## Prints the current state of the hypergraph
    def status(self):
        s = f"--- Status of {self._filename} ---\n"    + \
            f"Current iteration: {self._iteration}\n"  + \
            f"Edges of size k: {self._E_current}\n"    + \
            f"Auto-stop state: {self._stopping_arr}\n rw_index {self._rw_index} ; stopping sum {self._stopping_sum}\n" + \
            f"Total runtime: {self._runtime} ns or {self._runtime // 1000000} ms\n" + \
            f"--------------------------------\n"
        print(s)

    ###### outdated #######
    # ## Observation: as of this 28.11.2022 implementation, the algorithm tends to endlessly add 2-edges. This method cuts down on the number of repeated hyperedges to a given maximum number
    # ## returns a number (int) of all those pruned away

    # ## edit 1: or no, it doesn't now. What is going on? Now it's stabilizing around some values
    # #  edit 2: Think I've found an odd bug. After loading up the file into the object you'd normally need to use the method init_hypergraph() to set the initial state for the algorithm to run on. If you do this, the test graph windsurfers.gt will stabilize to some hypergraph of 800 2-edges. If you don't initialize and try to run the algorithm, it'll throw an error as per planned; but if you then initialize it and run the recovery algorithm, it will "stabilize" around 400 2-edges 430 3-edges. The 3-edges part is stable but the 2-edges part will endlessly rise in count.

    # def prune_hypergraph(self, threshold) -> int:

    #     assert type( threshold)  == int and threshold >= 0

    #     stats = 0
    #     for hyperedge in self._current_hypergraph:
    #         count = self._current_hypergraph[ hyperedge]
    #         if count > threshold:
    #             stats += count - threshold
    #             self._current_hypergraph[ hyperedge] = threshold
    #     print(f"{stats} hyperedges pruned")
    #     return stats