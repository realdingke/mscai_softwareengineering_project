import os
import os.path as osp

# summarises all absolute paths
#ROOT_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = osp.join(ROOT_PATH, '..') + '/dataset'  # /car_data

IMG_ROOT_PATH = DATA_PATH + '/images'

LABEL_PATH = DATA_PATH + '/labels_with_ids'

TRAIN_DATA_PATH = IMG_ROOT_PATH + '/train'

CFG_DATA_PATH = osp.join(ROOT_PATH, '/lib', '/cfg')







# summarises important relative paths
DATA_REL_PATH = '/dataset'


