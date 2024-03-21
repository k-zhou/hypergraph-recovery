import json
from test_helpers import *
test_dict = create_random_dict_fs()
print("The original dictionary is : " + str(test_dict))

# using encode() + dumps() to convert to bytes
# (!) This does not work with frozenset:s as keys
## TypeError: keys must be str, int, float, bool or None, not frozenset
# res_bytes = json.dumps(test_dict).encode('utf-8')
# transform this into 
 
# printing type and binary dict 
#print("The type after conversion to bytes is : " + str(type(res_bytes)))
#print("The value after conversion to bytes is : " + str(res_bytes))