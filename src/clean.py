import os
import os.path as osp
import paths
import pickle
import shutil


def clean_files_a():
    """
    Clean up all files that will be added to the original file in the second run
    """

    file_name_path = paths.PATHS_OBJ_PATH
    with open(file_name_path, 'rb') as f:
        paths_loader = pickle.load(f)
    # image_root = paths_loader.IMG_ROOT_PATH
    train_root = paths_loader.TRAIN_DATA_PATH
    label_root = paths_loader.LABEL_PATH
    client_data_path = paths.CLIENT_DATA_PATH
    seqs_name_path = osp.join(client_data_path, 'seqs_name_path.data')
    with open(seqs_name_path, 'rb') as f:
        seqs_name_dict = pickle.load(f)
    seqs = seqs_name_dict['empty_seqs'] + seqs_name_dict['labeled_seqs']

    # Delete gt and labels directories
    for seq in seqs:
        gt_path = osp.join(train_root, f'{seq}', 'gt')
        if osp.exists(gt_path):
            shutil.rmtree(gt_path)
        label_path = osp.join(label_root, f'train/{seq}/img1')
        if osp.exists(label_path):
            shutil.rmtree(label_path)

    # Clear image path directories
    image_path_dir = osp.join(paths.ROOT_PATH, 'data')
    for file in os.listdir(image_path_dir):
        file_path = osp.join(paths.ROOT_PATH, 'data', file)
        os.remove(file_path)


def clean_model():
    model_root = osp.join(paths.ROOT_PATH, '../exp/mot')
    # print(model_root)
    if osp.exists(model_root):
        for directories in os.listdir(model_root):
            dir_path = osp.join(model_root, directories)
            if directories != 'hrnet_pretrained':
                shutil.rmtree(dir_path)



