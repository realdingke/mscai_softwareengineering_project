from cord.client import CordClient  # pip install cord-client-python
import re
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
    pattern = '(?<=\w)\s(?=\w)'
    seqs = []
    try:
        for label_uid in project.get_labels_list():
            seq = client.get_label_row(label_uid)['data_title']
            try:
                seq = re.sub(pattern, '_', seq)
                seqs.append(seq)
            except:
                seq = seq
                seqs.append(seq)
        return seqs
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


def gen_obj_json(root='/content/drive/MyDrive/car_data_MCMOT/', client=load_cord_data()):
    project = client.get_project()
    pattern = '(?<=\w)\s(?=\w)'
    obj_jsons_list = []
    for label_uid in project.get_labels_list():
        label = client.get_label_row(label_uid)
        seq_name = label['data_title']
        try:
            seq_name = re.sub(pattern, '_', seq_name)
        except:
            seq_name = seq_name
        path = osp.join(root, seq_name)
        mkdirs(path)
        filename = path + '/objects.json'
        with open(filename, 'w') as f:
            json.dump(label, f)
        obj_jsons_list.append(label)
    return obj_jsons_list


def judge_video_info(obj_jsons_list):
    empty_seqs = []
    pattern = '(?<=\w)\s(?=\w)'
    for obj_json in obj_jsons_list:
        if len(list(obj_json["data_units"].values())[0]["labels"].keys()) <= 1 and \
          len(list(obj_json["data_units"].values())[0]["labels"]["0"]["objects"]) == 0: 
            seq_name = obj_json['data_title']
            try:
                seq_name = re.sub(pattern, '_', seq_name)
            except:
                seq_name = seq_name
            empty_seqs.append(seq_name)
    return empty_seqs
        
