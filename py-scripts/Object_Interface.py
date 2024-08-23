import os
from Hypergraph_Reconstructor import *
from fnmatch                  import fnmatch

class Object_Interface:

    def __init__(self) -> None:

        self._dataset_location = "../source-data/"
        self._output_location  = "../output/"
        self._container        = []
        self._current          = 0

        s = "-- Hypergraph Reconstruction from Network Data --\n" + \
        f"Taking datasets from the folder: \"{self._dataset_location}\" \n" + \
         "(Use change_dataset_location() to change this.)\n" + \
        f"Outputting to folder: \"{self._output_location}\" \n" + \
         "(Use change_output_location() to change this.)\n" + \
        f"> run load_file() to start\n" + \
        "> run load_all()  to load all .gt files in the dataset folder\n"
        print(s)

    def get_dataset_location(self) -> str:
        return self._dataset_location
    
    def get_output_location(self) -> str:
        return self._output_location
    
    def change_dataset_location(self, new=None) -> None:
        if new == None: new = input("Enter the new path to the datasets folder:\n")
        self._dataset_location = new

    def change_output_location(self, new=None) -> None:
        if new == None: new = input("Enter the new path to the outputs folder:\n")
        self._output_location = new

    def get_container_len(self) -> int:
        return len(self._container)

    def load_file(self, filename=None) -> None:
        if filename == None: filename = input(f"Enter the file / path to file. Current location is\n" + \
                                              f"{self._dataset_location}")
        self._container.append( Hypergraph_Reconstructor( self._dataset_location + filename) )

    # loads all .gt files in the DATASET LOCATION folder to self._container
    def load_all(self) -> None:
        file_list = []
        len_before = len(self._container)
        counter = 0
        for filename in os.listdir(self._dataset_location):
            if fnmatch(filename, "*.gt"):
                file_list.append(filename)
        for item in file_list:
            print(f"[{len_before + counter}] {item}")
            self.load_file(item)
            counter += 1
        print(f"{len(file_list)} files appended to self._container[{len_before}+{len(file_list)}]")

    def set_current(self,  target=None ) -> bool:
        if not target == None and 0 <= target and target < len(self._container):
            self._current = target
            print(f"---- ---- ---- ----\nSELECTED [{self._current}] : {self._container[self._current]._filename_only}")
            return True
        else:
            print(f"CANNOT SELECT [{target}]")
            return False

    # runs for a set amount of iterations default 100
    def run_algorithm(self, max_iterations=None, autostop=None, min_iterations=None) -> None:
        if min_iterations == "again":
            n = 2*self._container[self._current]._iteration +1
            print(f"Running again for a minimum of {n} iterations.")
            min_iterations = n
        elif not min_iterations == None:
            min_iterations = int(min_iterations)
        else:
            min_iterations = 100
        if len(self._container) > 0:
            print(f"Running on self._container[{self._current}]. Target {max_iterations} (default 100) iterations.")
            self._container[self._current].run_algorithm( max_iterations, autostop, min_iterations)
            print(f"Stopped")
        else:
            print("No files loaded.")

    #
    def run_until_autostopped(self, min_iterations=None ) -> None:
        r1 = self._container[self._current]
        while not r1._auto_stopped:
            self.run_algorithm(10000, True, min_iterations)
            r1.status()
        sleep(1)
        r1._auto_stopped = False

    #
    def run_forced(self, iterations) -> None:
        r1 = self._container[self._current]
        self.run_algorithm(iterations, False)
        r1.status()
        self.save_output()

    # saves the log to file
    def save_output(self) -> None:
        rec = self._container[self._current]
        fn  = self._output_location + rec._filename_only + "(log)" + ".txt"
        rec.output_to_log(fn)
        print(f"Saved to {fn}")
        return

    def save_history(self) -> None:
        rec = self._container[self._current]
        fn  = self._output_location + rec._filename_only + "(history)" + ".txt"
        rec.output_history_to_log(fn)
        print(f"Saved to {fn}")
        return

    def save_history_exact(self) -> None:
        rec = self._container[self._current]
        fn  = self._output_location + rec._filename_only + "(history_exact)" + ".txt"
        rec.output_history_exact_to_log(fn)
        print(f"Saved to {fn}")
        return

    def save_hypergraph(self) -> None:
        rec = self._container[self._current]
        fn  = self._output_location + rec._filename_only + "(h_graph)" + ".txt"
        rec.output_hypergraph_to_log(fn)
        print(f"Saved to {fn}")
        return