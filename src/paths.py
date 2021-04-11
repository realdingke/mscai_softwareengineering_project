import os
import os.path as osp

# summarises all absolute paths
#ROOT_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = osp.join(ROOT_PATH, '..') + '/dataset'

IMG_ROOT_PATH = DATA_PATH + '/images'

LABEL_PATH = DATA_PATH + '/labels_with_ids'

TRAIN_DATA_PATH = IMG_ROOT_PATH + '/train'

CFG_DATA_PATH = osp.join(ROOT_PATH, '/lib', '/cfg')

OUTPUTS_PATH = DATA_PATH + '/outputs'

RESULTS_PATH = DATA_PATH + '/results'





# summarises important relative paths
DATA_REL_PATH = '/dataset'


