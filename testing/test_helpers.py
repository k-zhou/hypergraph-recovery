from random import random, randint

# create a dict with a few randomly chosen frozensets mapping to 1
def create_random_dict_fs():
    out = dict()
    for i in range(0, randint(1,5)):
        s  = set()
        for j in range(0, randint(3,6)):
            s.add(randint(1,30))
        #print(s, frozenset(s))
        out[frozenset(s)] = 1
    return out