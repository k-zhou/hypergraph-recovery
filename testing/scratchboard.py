adj = dict()
# use frozenset to use as keys in dict, note that frozensets are immutable sets
adj[frozenset({1,2,5})] = 1
nmax = 10
for i in range(0,nmax):
    for j in range(i+1,nmax):
        for k in range(j+1,nmax):
            print("The key {%d %d %d } gives value" % (i, j, k), adj.get(frozenset({i, j, k}), 0) )

# Projection component
# given a hyperedge E_h, mark all nodes it connects as pairwise connected in G
# -> how to iterate a python set?
#   <- convert to list

s1 = frozenset([1,2,5,7])
s2 = frozenset([6,4,9,3])

# get unique pairs
sL2 = list(s2)
for i in range(0, len(sL2)):
  for j in range(i+1, len(sL2)):
   subset_ = frozenset([sL2[i], sL2[j]])
   print(subset_, subset_ < s2)

#  this holds all of the hyperedges in the hypergraph H
# the entries are of the type frozenset([u_int , ...]) mapped to value 1
h_edges = dict()


# can you compare dicts?
d1 = dict()
d1[frozenset([1,3,5,7])] = 1
d2 = dict()
d2[frozenset([1,3,5,7])] = 1
d1 == d2
d1[frozenset([2,3,5,7])] = 1
d1 == d2
d2[frozenset([7,5,2,3])] = 1
d1 == d2
# yes you can
