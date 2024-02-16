

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
    last_ = 0
    last_ = len(fname) -1
    for i in range(0, len(fname)):
        j = len(fname) -1 -i
        if fname[j] == ch:
            last_ = j # assuming the path and file is valid

    stripped = fname[0:last_]
    return stripped




# def print_list_magnitudes(list, end = None):
#     if end != None:
#         assert type( end) == str
#     for element in list:
#
#         print(f"{element}, ", end = '')
#     print("", end = end)
