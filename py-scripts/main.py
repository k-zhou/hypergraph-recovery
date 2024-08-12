### source ######################################
# arXiv:2008.04948v3 [cs.SI] 25. May 2021
# Hypergraph reconstruction from network data
# J-G. Young, G. Petri, T. P. Peixoto

# https://docs.python.org/3/library/math.html
# https://graph-tool.skewed.de/static/doc/graph_tool.html
# math.comb(n,k)    # binomial coefficient
# math.factorial(x)

# import string

from Object_Interface import *
from project_tests   import *

### Main ##########

def main() -> None:
    if not "Reconstructor" in globals():
        global Reconstructor
        print("Creating object \"Reconstructor\" ...")
        Reconstructor = Object_Interface()
    else:
        print("Needs variable \"Reconstructor\" to be available to start.")

    return

def help():
    print(f"FILE_LOADED, DATASET_LOCATION, OUTPUT_LOCATION, RECONSTRUCTORS, CURRENT")
    print(f"load_file(filename = None) ")
    print(f"load_all() ")
    print(f"run_algorithm( max_iterations = None ) ")

# Running this file will call the main() function like an entrypoint as per convention familiar to C programs
if __name__ == "__main__":
    main()

