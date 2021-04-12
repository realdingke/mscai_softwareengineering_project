import os
import os.path as osp
import pickle

# summarises all absolute paths
#ROOT_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# summarises important relative paths
DATA_REL_PATH = '/dataset'


class paths_loader(object):
    """The project paths loader
    """
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_REL_PATH = '/dataset'
    def __init__(self, file_name_path):
        with open(file_name_path, 'rb') as f:
            path_info_dct = pickle.load(f)
        
        self.DATA_PATH = osp.join(paths_loader.ROOT_PATH, '..', 'dataset', f"{path_info_dct['pn']}")

        self.IMG_ROOT_PATH = self.DATA_PATH + '/images'

        self.LABEL_PATH = self.DATA_PATH + '/labels_with_ids'

        self.TRAIN_DATA_PATH = self.IMG_ROOT_PATH + '/train'

        self.CFG_DATA_PATH = osp.join(paths_loader.ROOT_PATH, 'lib', 'cfg')
        
        self.DS_JSON_PATH= osp.join(paths_loader.ROOT_PATH, '..', 'dataset')

        self.OUTPUTS_PATH = self.DATA_PATH + '/outputs'

        self.RESULTS_PATH = self.DATA_PATH + '/results'
    
    def __str__():
        return f"This is the project path loader class"
        
