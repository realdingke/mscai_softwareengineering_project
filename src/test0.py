from src.download import mkdir

def make_file(filename, dct):
    with open(filename, 'w') as f:
        json.dump(dct, f)

def print_array(array=np.array([1,2,3,4])):
    print(np.array(array))