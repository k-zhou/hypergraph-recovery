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

### Main ##########

# plan:
## load files from pre-determined source files folder
## output results to txt file, including
##  runtime, initial and final h-graph h-edges count,

def main():

    # global variables 
    global help
    global FILE_LOADED, DATASET_LOCATION, OUTPUT_LOCATION, RECONSTRUCTORS, CURRENT
    FILE_LOADED        = False
    DATASET_LOCATION   = "../source-data/"
    OUTPUT_LOCATION    = "../output/"
    RECONSTRUCTORS     = []
    CURRENT            = 0
    #help               = help()

    print("\n-- Hypergraph Reconstruction from Network Data --")
    print(f"Taking datasets from the folder: \"{DATASET_LOCATION}\" \nModify global variable DATASET_LOCATION to change this.")
    print(f"Outputting to folder: \"{OUTPUT_LOCATION}\" \nModify global variable OUTPUT_LOCATION to change this.")
    print("> run load_file() to start")
    print("> run load_all()  to load all .gt files in the dataset folder")
    #hypergraph   = init_hypergraph(G_orig)
    #graph_order  = len(list(G_orig.vertex_index))
    #results      = None

    return

#

def load_file(filename = None):
    global DATASET_LOCATION, RECONSTRUCTORS
    if filename == None:
        filename      = input(f"Enter the file / path to file: {DATASET_LOCATION}")
        RECONSTRUCTORS.append( Hypergraph_Reconstructor( DATASET_LOCATION + filename) )
        #print(f"Loaded file { filename} to object \"RECONSTRUCTORS[{ len(RECONSTRUCTORS) - 1 }]\"")
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

def set_current( target = None ) -> bool:
    global RECONSTRUCTORS
    if not target == None and 0 <= target and target < len(RECONSTRUCTORS):
        global CURRENT
        CURRENT = target
        print(f"SELECTED [{CURRENT}] : {RECONSTRUCTORS[CURRENT]._filename_only}")
        return True
    else:
        print(f"CANNOT SELECT [{target}]")
        return False

# runs for a set amount of iterations default 100
def run_algorithm( max_iterations = None, autostop = None, min_iterations = None):
    global RECONSTRUCTORS, CURRENT
    if len(RECONSTRUCTORS) > 0:
        print(f"Running on RECONSTRUCTORS[{CURRENT}]. Target {max_iterations} (default 100) iterations.")
        RECONSTRUCTORS[CURRENT].run_algorithm( max_iterations, autostop, min_iterations)
        print(f"Stopped")
    else:
        print("No files loaded.")

#
def run_until_autostopped( min_iterations = None ) -> None:
    global RECONSTRUCTORS, CURRENT
    r1 = RECONSTRUCTORS[CURRENT]
    while not r1._auto_stopped:
        run_algorithm(10000, True, min_iterations)
        r1.status()
    sleep(1)
    r1._auto_stopped = False

#
def run_forced(iterations) -> None:
    global RECONSTRUCTORS, CURRENT
    r1 = RECONSTRUCTORS[CURRENT]
    run_algorithm(iterations, False)
    r1.status()
    save_output()

# saves the log to file
def save_output() -> None:
    global RECONSTRUCTORS, CURRENT, OUTPUT_LOCATION
    rec = RECONSTRUCTORS[CURRENT]
    fn  = OUTPUT_LOCATION + rec._filename_only + "(log)" + ".txt"
    rec.output_to_log(fn)
    print(f"Saved to {fn}")
    return

def save_history() -> None:
    global RECONSTRUCTORS, CURRENT, OUTPUT_LOCATION
    rec = RECONSTRUCTORS[CURRENT]
    fn  = OUTPUT_LOCATION + rec._filename_only + "(history)" + ".txt"
    rec.output_history_to_log(fn)
    print(f"Saved to {fn}")
    return

def save_hypergraph() -> None:
    global RECONSTRUCTORS, CURRENT, OUTPUT_LOCATION
    rec = RECONSTRUCTORS[CURRENT]
    fn  = OUTPUT_LOCATION + rec._filename_only + "(h_graph)" + ".txt"
    rec.output_hypergraph_to_log(fn)
    print(f"Saved to {fn}")
    return

def help():
    print(f"FILE_LOADED, DATASET_LOCATION, OUTPUT_LOCATION, RECONSTRUCTORS, CURRENT")
    print(f"load_file(filename = None) ")
    print(f"load_all() ")
    print(f"run_algorithm( max_iterations = None ) ")

# Running this file will call the main() function like an entrypoint as per convention familiar to C programs
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
