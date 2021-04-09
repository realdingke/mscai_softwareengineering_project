from cord.client import CordClient  # pip install cord-client-python

import os
import os.path as osp
import json

class ClientError(Exception):
    pass


def mkdirs(d):
    """make dir if not exist"""
    if not osp.exists(d):
        os.makedirs(d)


def load_cord_data(project_id='eec20d90-c014-4cd4-92ea-72341c3a1ab5',
                   api_key='T7zAcCv2uvgANe4JhSPDePLMTTf4jN-hYpXu-XdMXaQ'):
    try:
        client = CordClient.initialise(
            project_id,  # Project ID of car
            api_key  # API key
        )
        # Get project info (labels, datasets)
        project = client.get_project()

        return client

    except:
        raise ClientError('Unable to load, please check the project id and API key')


def gen_seq_name_list(client):
    """
    returns a list of all seq_names present in the current project
    """
    project = client.get_project()

    try:
        return [client.get_label_row(label_uid)['data_title'] for label_uid in project.get_labels_list()]
    except:
        return None


def get_cls_info(root='/content/drive/MyDrive/car_data_MCMOT/images/train/'):
    """load the saved class dict info"""
    with open(osp.join(root, 'cls2id.json'), 'r') as f:
        cls2id_dct = json.load(f)
    with open(osp.join(root, 'id2cls.json'), 'r') as f:
        id2cls_dct = json.load(f)
        id2cls_dct_new = {}
        for i in id2cls_dct.keys():
            id2cls_dct_new[int(i)] = id2cls_dct[i]

    return cls2id_dct, id2cls_dct_new
