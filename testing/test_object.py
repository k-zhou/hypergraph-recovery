class Tester:

    def __init__(self, filename):

        try:
            f = open(filename, "rt")
        except:
            print(f"Error in creating {filename}")
        else:
            f.close()
        self._filename = filename
        self._filename_pathless = strip_filename_path(self._filename)
        self._filename_only     = strip_filename_suffix(self._filename_pathless)
        self._path     = strip_filename_suffix(self._filename, '/')
        self._log      = []

        self._log.append(self._filename)
        self._log.append(self._filename_pathless)
        self._log.append(self._filename_only)
        self._log.append(self._path)


    def __del__(self):
        print("DEL")

def strip_filename_path(fname, ch = None):
    ch    = ch
    if ch == None:
        ch = '/'
    last_ = 0
    for i in range(0, len(fname)):
        j = len(fname) -1 -i
        if fname[j] == ch:
            last_ = j # assuming the path and file is valid
            break
    stripped = fname[last_ + 1: ]
    return stripped

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
            break
    stripped = fname[0:last_]
    return stripped

def run_test():
    PATH       = "testfolder/"
    FILENAME   = "test_file_name2"
    FILESUFFIX = ".txt"

    global t
    t = Tester(PATH + FILENAME + FILESUFFIX)
    print(t._log)
