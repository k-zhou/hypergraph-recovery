### source ######################################
# arXiv:2008.04948v3 [cs.SI] 25. May 2021
# Hypergraph reconstruction from network data
# J-G. Young, G. Petri, T. P. Peixoto

# https://docs.python.org/3/library/math.html
# https://graph-tool.skewed.de/static/doc/graph_tool.html
# math.comb(n,k)    # binomial coefficient
# math.factorial(x)

# import string
from Hypergraph_Reconstructor import *
from project_tests   import *
import os
from fnmatch         import fnmatch

# global variables declared under main()
#
def load_file(filename = None):
    global DATASET_LOCATION, RECONSTRUCTORS
    if filename == None:
        filename      = input(f"Enter the file / path to file: {DATASET_LOCATION}")
        RECONSTRUCTORS.append( Hypergraph_Reconstructor( DATASET_LOCATION + filename) )
        print(f"Loaded file { filename} to object \"RECONSTRUCTORS[{ len(RECONSTRUCTORS) - 1 }]\"")
    else:
        RECONSTRUCTORS.append( Hypergraph_Reconstructor( DATASET_LOCATION + filename) )

# loads all .gt files in the DATASET LOCATION folder to RECONSTRUCTORS
def load_all():
    global DATASET_LOCATION, RECONSTRUCTORS
    file_list = []
    len_before = len(RECONSTRUCTORS)
    counter = 0
    for filename in os.listdir(DATASET_LOCATION):
        if fnmatch(filename, "*.gt"):
            file_list.append(filename)
    for item in file_list:
        print(f"[{len_before + counter}] {item}")
        load_file(item)
        counter += 1
    print(f"{len(file_list)} files appended to RECONSTRUCTORS[{len_before}+{len(file_list)}]")

# runs for a set amount of iterations default 100
def run_algorithm( max_iterations = None):
    global RECONSTRUCTORS, CURRENT
    if len(RECONSTRUCTORS) > 0:
        print(f"Running on RECONSTRUCTORS[{CURRENT}]")
        RECONSTRUCTORS[CURRENT].run_algorithm( max_iterations)
    else:
        print("No files loaded.")

# TODO: output to file
# saves the log to file
def save_output():
    global RECONSTRUCTORS, OUTPUT_LOCATION

    return


### Main ##########

# plan:
## load files from pre-determined source files folder

## timestamp
## run algorithm until h-edge count differences plateau
## timestamp

## output results to txt file, including
##  runtime, initial and final h-graph h-edges count,

def main():
    FILE_LOADED        = False
    DATASET_LOCATION   = "../source-data/"
    OUTPUT_LOCATION    = "../output/"
    RECONSTRUCTORS     = []
    CURRENT            = 0
    print("\n-- Hypergraph Reconstruction from Network Data --")
    print(f"Taking datasets from the folder: \"{DATASET_LOCATION}\" \nModify global variable DATASET_LOCATION to change this.")
    print(f"Outputting to folder: \"{OUTPUT_LOCATION}\" \nModify global variable OUTPUT_LOCATION to change this.")
    print("> run load_file() to start")
    print("> run load_all()  to load all .gt files in the dataset folder")
    #hypergraph   = init_hypergraph(G_orig)
    #graph_order  = len(list(G_orig.vertex_index))
    #results      = None

    return

if __name__ == "__main__":
    main()
"""

# run the algorithm

FILENAME = "../source_data/windsurfers.gt"
G_orig   = init( FILENAME )
if (G_orig == None): raise Exception("Cannot continue with null as the graph G_orig.")
print("Loaded file", FILENAME, "\nrunning algorithm...")
try:
    result = find_best_hypergraph( G_orig)
except:
    print("Error.")

# Visualisation

stop_execution = input("Algorithm finished. Press any key to exit...")
 """
