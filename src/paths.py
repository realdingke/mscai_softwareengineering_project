import os
import os.path as osp

# import pickle

# summarises all absolute paths
# ROOT_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


# summarises important relative paths
DATA_REL_PATH = '/dataset'


CLIENT_DATA_PATH = osp.join(ROOT_PATH + '/..' + DATA_REL_PATH, 'client_data')
        
PATHS_OBJ_PATH = osp.join(
    CLIENT_DATA_PATH,
    'path_names_obj.data',
)   # stores the paths_loader object save path


class paths_loader(object):
    """The project paths loader
    """
    ROOT_PATH = ROOT_PATH
    DATA_REL_PATH = DATA_REL_PATH

    def __init__(self):

        self.DATA_PATH = osp.join(paths_loader.ROOT_PATH, '..', 'dataset')

        self.IMG_ROOT_PATH = None

        self.LABEL_PATH = None

        self.TRAIN_DATA_PATH = None

        self.CFG_DATA_PATH = osp.join(paths_loader.ROOT_PATH, 'lib', 'cfg')

        self.DS_JSON_PATH = osp.join(paths_loader.ROOT_PATH, '..', 'dataset')

        self.LOSS_CURVES_PATH = None

        self.TEST_DIR_NAME_PATH = None

        self.SEQS_NAME_PATH = None

        self.RESULTS_PATHS = None

    def update(self):

        self.IMG_ROOT_PATH = self.DATA_PATH + '/images'

        self.LABEL_PATH = self.DATA_PATH + '/labels_with_ids'

        self.TRAIN_DATA_PATH = self.IMG_ROOT_PATH + '/train'

        self.CFG_DATA_PATH = osp.join(paths_loader.ROOT_PATH, 'lib', 'cfg')

        self.DS_JSON_PATH = osp.join(paths_loader.ROOT_PATH, '..', 'dataset')

        self.OUTPUTS_PATH = self.DATA_PATH + '/outputs'

        self.RESULTS_PATH = self.DATA_PATH + '/results'

        self.LOSS_CURVES_PATH = self.DATA_PATH + '/Loss_Figure'

        self.TEST_DIR_NAME_PATH = self.IMG_ROOT_PATH + '/test/'
        
    def __str__(self):
        return f"This is the project path loader class"
