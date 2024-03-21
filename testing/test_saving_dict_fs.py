from test_helpers import *

# Instead of saving a dict of frozensets as a binary file, transform it into a list of text in the format

# A1 A2 A3 ... AN XX

# where A1 .. AN are members of the frozen/set which was the key of the dict,
# and XX the value mapped to the key.
# Then save the hypergraph as a text file with each row as a hyperedge
# This implies that each row must have a minimum of 2 integers, in case of data verification.

D = create_random_dict_fs()
print(f"Original dict: \n{D}")

filename = 'testfolder/test_save_.txt'
with open(filename, mode='wt', encoding="utf-8") as fw:
    fw.write("\\\ THIS FILE GETS OVERWRITTEN ON EVERY TESTING RUN. DO NOT SAVE TO THIS FILE. ///\n")
    for pair in D.items():
        for element in pair[0]:
            fw.write(str(element) + ' ')
        fw.write(str(pair[1]) + '\n')

D2 = dict()
with open(filename, mode='rt', encoding="utf-8") as fr:
    line = fr.readline()
    # parse line and convert back to hyperedge
    while(line):
        split = line.split(' ')
        s = set()
        for element in split:
            try: 
                s.add(int(element))
            except:
                s = set()
                break
        if len(s) >= 2:
            n = s.pop()
            D2[frozenset(s)] = n
        line = fr.readline()

print(f"Read dict: \n{D2}")
