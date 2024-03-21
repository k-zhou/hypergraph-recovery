from time      import struct_time, strftime
from random    import random
from itertools import chain

PATH     = "testfolder/"
FILESUFFIX = ".txt"
FILENAME = "test_file_name"
ALT_N    = 1

#########################################
# https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
# It recommends using the with keyword
# with open('workfile', encoding="utf-8") as f:
#     read_data = f.read()

#########################################

# create new file to write to
# if filename taken, iterate through alternative names

def reset_alt_suff():
    global ALT_N
    ALT_N = 1
    return

def get_alt_suff():
    global ALT_N
    ALT_N += 1
    return " (" + str(ALT_N) + ")"

def get_date_custom():
    # formatting listed at
    # https://docs.python.org/3/library/time.html#time.strftime
    # alt with seconds:  strftime("%Y%m%d-%H%M%S")
    return strftime("-%Y%m%d-%H%M")

def get_date_custom_s():
    return strftime("%Y%m%d-%H%M%S")

def handle_file():

    global file, tryname
    try:
        # open file
        tryname = PATH + FILENAME + get_date_custom() + FILESUFFIX
        print(tryname)
        file    = open(tryname, "xt")
    except:
        # error
        # find alternate names
        while True:
            try:
                tryname = PATH + FILENAME + get_date_custom() + get_alt_suff() + FILESUFFIX
                print(tryname)
                file    = open(tryname, "xt")
            except:
                print("err name also taken: " + str(ALT_N))
                continue
            else:
                # print("success " + str(ALT_N))
                print("Created file with alternate name.")
                reset_alt_suff()
                break
    else:
        # close file
        print("Successfully created requested file.")
    finally:
        # regardless
        file.write("Success at " + strftime("%H%M%S"))

        file.close()

def create_file():

    global file, newname
    try:
        newname = PATH + FILENAME + get_date_custom() + FILESUFFIX
        file = open(newname, "xt")
    except:
        while True:
            try:
                newname = PATH + FILENAME + get_date_custom() + get_alt_suff() + FILESUFFIX
                file = open(newname, "xt")
            except:
                continue
            else:
                reset_alt_suff()
                break
    finally:
        file.close()

    return newname

def write_data():

    global file
    try:
        fname = PATH + FILENAME + FILESUFFIX
        file = open(fname, "at")
    except:
        print("")
    else:
        alldata = []
        
        list1    = [random() for item in range(0,5)]
        list2   = [ [random() for item in range(0,2)] for item in range(0,3) ]
        list2   = list(chain.from_iterable(list2))
        
        alldata.append(get_date_custom_s())
        alldata.append(list1)
        alldata.append(list2)
        
        for item in alldata:
            file.write(str(item) + '\n')
    finally:
        file.close()
    return
