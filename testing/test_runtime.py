import time

# This experiments with the time module
def test_process_time():
    start_time = time.process_time_ns() # int
    print(f"Starting time marked. It is {start_time}")

    n = 0
    for i in range(0,20):
        n += i
    end_time = time.process_time_ns()
    print(f"It is now {end_time}")
    print(f"{end_time - start_time} has elapsed")

    input("Press any key to exit.")

def test_time():
    start_time = time.time_ns()
    print(f"Starting time marked. It is {start_time}")

    n = 0
    for i in range(0,20):
        n += i

    end_time   = time.time_ns()
    print(f"It is now {end_time}")
    print(f"{end_time - start_time} has elapsed")

    input("Press any key to exit.")

def test_runtime(func = None):
    if func == None:
        print("No timing function to test.")
        return
    
    start_time = func()
    print(f"Starting time marked. It is {start_time}")

    n = 0
    for i in range(0,20):
        n += i

    end_time   = func()
    print(f"It is now {end_time}")
    print(f"{end_time - start_time} has elapsed")

    input("Press any key to exit.")
