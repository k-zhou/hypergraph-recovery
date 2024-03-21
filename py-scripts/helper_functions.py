from time      import struct_time, strftime

ALT_FILE_SUFFIX_NUMBER = 1

### maths methods
# helper: calculate  a! / b!
def factorial_div_factorial(top, bot):
    assert type(top) == int
    assert type(bot) == int
    assert top >= 0
    assert bot >= 0
    if top > bot:
        i   = top
        ans = 1
        while ( i > bot ):
            ans *= i
            i   -= 1
        return ans
    elif top == bot:
        return 1
    else:
        return 1/factorial_div_factorial(bot, top)

### String methods
# helper: removes the path to the file from the filename
def strip_filename_path(fname, ch = None):
    ch    = ch
    if ch == None:
        ch = '/'
    last_ = 0
    for i in range(0, len(fname)):
        j = len(fname) -1 -i
        if fname[j] == ch:
            last_ = j # assuming the path and file is valid

    stripped = fname[last_ + 1: ]
    return stripped

# helper: removes the file type suffix from the pathless filename
def strip_filename_suffix(fname, ch = None):
    ch    = ch
    if ch == None:
        ch = '.'
    last_ = len(fname) -1
    for i in range(0, len(fname)):
        j = len(fname) -1 -i
        if fname[j] == ch:
            last_ = j # assuming the path and file is valid

    stripped = fname[0:last_]
    return stripped
# helper: only keeps the suffix including the dot
def get_filename_suffix(fname, ch = None):
    ch    = ch
    if ch == None:
        ch = '.'
    last_ = len(fname) -1
    for i in range(0, len(fname)):
        j = len(fname) -1 -i
        if fname[j] == ch:
            last = j
    stripped = fname[last_:]
    return stripped
    
### Dates
#
def get_date_custom():
    # formatting listed at
    # https://docs.python.org/3/library/time.html#time.strftime
    return strftime("-%Y%m%d-%H%M")

# with seconds at the end
def get_date_custom_s():
    return strftime("%Y%m%d-%H%M%S")

### File handling
#
def reset_alt_suff():
    global ALT_FILE_SUFFIX_NUMBER
    ALT_FILE_SUFFIX_NUMBER = 1
    return
#
def get_alt_suff():
    global ALT_FILE_SUFFIX_NUMBER
    ALT_FILE_SUFFIX_NUMBER += 1
    return " (" + str(ALT_FILE_SUFFIX_NUMBER) + ")"
#
def write_to_file(fname, data):
    filename   = strip_filename_suffix(fname)
    filesuffix = get_filename_suffix(fname)
    data    = data
    global f, tryname
    try:
        tryname = filename + get_date_custom() + filesuffix
        file    = open(tryname, "xt")
    except:
        while True:
            try:
                tryname = filename + get_date_custom() + get_alt_suff() + filesuffix
                file    = open(tryname, "xt")
            except:
                continue
            else:
                reset_alt_suff()
                break
    finally:
        file.write(data)
        file.close()

################################

# def print_list_magnitudes(list, end = None):
#     if end != None:
#         assert type( end) == str
#     for element in list:
#
#         print(f"{element}, ", end = '')
#     print("", end = end)
