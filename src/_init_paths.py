import os.path as osp
import sys

from src import paths

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

def gen_init_data_path(project_name):
    """generate the data save path, called at the beginning of project
    """
    paths.DATA_PATH = osp.join(paths.DATA_PATH, project_name)
#    paths.DATA_PATH  = osp.join(osp.join(paths.ROOT_PATH, '..') + paths.DATA_REL_PATH, project_name)
