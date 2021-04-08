import os.path as osp
import os
import numpy as np
import json

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def make_file(filename, dct):
    with open(filename, 'w') as f:
        json.dump(dct, f)

def print_array(array=np.array([1,2,3,4])):
    print(np.array(array))