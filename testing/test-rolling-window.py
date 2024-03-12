from random import randint, choice

class Test_Rolling_Window:
    
    def __init__(self, window_size = None) -> None:

        self._window_size = 7 # test size, modify to convenience
        if not window_size == None:
            self._window_size = window_size
        self._window = [ (0, 0) for i in range(0, self._window_size) ]
        self._index  = 0

        self._arr_w  = 5
        self._summed_arr = [ 0 for i in range(0, self._arr_w) ]
    
    def advance(self, steps = None) -> None:

        _steps = 1
        if not steps == None:
            _steps = steps
        
        for i in range(0, _steps):
            # generate random data to use
            next_data = ( choice([-1, 1]), randint(0, self._arr_w -1) )
            # test the feature
            print(self._window)
            print(self._summed_arr)

            tup       = self._window[self._index]
            print(f"Index currently at {self._index} pointing at {tup}")
            self._summed_arr[tup[1]] -= tup[0]

            tup       = next_data
            self._window[self._index] = tup
            print(f"Next data: {next_data}")
            self._summed_arr[tup[1]] += tup[0]

            print(self._window)

            print(self._summed_arr, "\n")

            #wrap around
            self._index += 1
            if self._index >= self._window_size:
                self._index = 0