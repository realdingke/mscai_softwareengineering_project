from gen_labels import load_cord_data

import os.path as osp
import os
import numpy as np
import json
import re
import uuid
from datetime import datetime
from cord.utils import label_utils
from cord.client import CordClient  # ! pip install cord-client-python


COLORS = [
    '#D33115',
    '#1979a9',
    '#e07b39',
    '#edb06b',
    '#69bdd2',
    '#80391e',
    '#1c100b',
    '#ebdab4',
    '#042f66',
    '#b97455',
    '#44bcd8',
]


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def read_det_results(result_path):
    results = []
    with open(result_path, 'r') as f:
        for line in f.readlines():
            results.append(line.split(','))
    return results


def gen_new_obj(tid, 
                x, 
                y, 
                w, 
                h, 
                conf, 
                clsid, 
                seq_width, 
                seq_height,
                id2color_dict,
                tid2objhash_dct, 
                id2cls_dict, 
                featureHash_dct,
                creator_email='grouproject851@gmail.com', 
                gmt_format='%a, %d %b %Y %H:%M:%S UTC'):
    data_dct = {}

    data_dct['name'] = id2cls_dict[clsid][0].capitalize() + \
                       id2cls_dict[clsid][1:]
    data_dct['color'] = id2color_dict[clsid]
    data_dct['shape'] = 'bounding_box'
    data_dct['value'] = id2cls_dict[clsid]
    data_dct['createdAt'] = datetime.utcnow().strftime(gmt_format)
    data_dct['createdBy'] = creator_email
    data_dct['confidence'] = float(conf)
    data_dct['objectHash'] = tid2objhash_dct[tid]
    data_dct['featureHash'] = featureHash_dct[clsid]
    data_dct['manualAnnotation'] = False

    data_dct['boundingBox'] = {}
    data_dct['boundingBox']['x'] = float(x) / seq_width  # obj
    data_dct['boundingBox']['y'] = float(y) / seq_height
    data_dct['boundingBox']['w'] = float(w) / seq_width
    data_dct['boundingBox']['h'] = float(h) / seq_height

    return data_dct


def upload_results(client, 
                   results, 
                   root_path='/content/drive/MyDrive/cattle_data/images/train/',
                   creator_email='grouproject851@gmail.com', 
                   gmt_format='%a, %d %b %Y %H:%M:%S UTC'):
    """"""

    id2cls_dct_path = osp.join(root_path, 'id2cls.json')
    featureHash_dct_path = osp.join(root_path, 'featureHash.json')
    with open(id2cls_dct_path, 'r') as f:
        id2cls_dict = json.load(f)
    with open(featureHash_dct_path, 'r') as f:
        featureHash_dct = json.load(f)
    id2color_dict = dict()
    for idx, clsid in enumerate(id2cls_dict.keys()):
        id2color_dict[clsid] = COLORS[idx]
    project = client.get_project()
    for label_uid in project.get_labels_list():
        label = client.get_label_row(label_uid)
        seqs_str = label['data_title']
        ##swap all space in between the seq_name to '_'
        pattern = '(?<=\w)\s(?=\w)'
        try:
            seqs_str = re.sub(pattern, '_', seqs_str)
        except:
            seqs_str = seqs_str
        tid2objhash_dct_path = osp.join(
            root_path, seqs_str, 
            f"{seqs_str}_tid2objhash.json",
        )
        with open(tid2objhash_dct_path, 'r') as f:
            tid2objhash_dct = json.load(f)
        seq_ini_file = osp.join(root_path, f"{seqs_str}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(
            seq_info[seq_info.find('imWidth=') + 
            8:seq_info.find('\nimHeight')]
        )
        seq_height = int(
            seq_info[seq_info.find('imHeight=') + 
            9:seq_info.find('\nimExt')]
        )
        deleted_frames = set()
        for frame, tid, x, y, w, h, conf, clsid, _ in results:
            if tid not in tid2objhash_dct.keys():
                random_objhash = str(uuid.uuid4())[:8]
                tid2objhash_dct[tid] = random_objhash
            try:
                obj_cls_dct = list(label['data_units'].values())[0]['labels'][str(int(frame) - 1)]
            except KeyError:
                list(label['data_units'].values())[0]['labels'][str(int(frame) - 1)] = {
                    'objects': [],
                    'classifications': [],
                }
            if str(int(frame) - 1) not in deleted_frames:
                deleted_frames.add(str(int(frame) - 1))
                list(label['data_units'].values())[0]['labels'][str(int(frame) - 1)]['objects'] = []
            data_dct = gen_new_obj(
                tid, 
                x, 
                y, 
                w, 
                h, 
                conf, 
                clsid, 
                seq_width, 
                seq_height,
                id2color_dict,
                tid2objhash_dct, 
                id2cls_dict,
                featureHash_dct, 
                creator_email=creator_email, 
                gmt_format=gmt_format,
            )
            list(label['data_units'].values())[0]['labels'][str(int(frame) - 1)]['objects'].append(data_dct)
        updated = label_utils.construct_answer_dictionaries(label)
        for obj_hash in tid2objhash_dct.values():
            if obj_hash not in updated['object_answers'].keys():
                updated['object_answers'][obj_hash] = {
                    'objectHash': obj_hash, 
                    'classifications': [],
                }
        client.save_label_row(label_uid, updated)


if __name__ == '__main__':
    project_id = '235aa1ec-8d5e-4253-b673-1386af826fae' # Project ID of drone
    api_key = 'vV_rHH11febK3F2ivQYO_qzlLO9nNTCPxaGblNrfJzg'
    result_path = '/content/drive/MyDrive/cattle_data/images/' + \
                  'results/cattle_dla_20/Video_of_cattle_1.mp4.txt'
    root_path = '/content/drive/MyDrive/cattle_data/images/train/'
    creator_email = 'grouproject851@gmail.com'
    gmt_format = '%a, %d %b %Y %H:%M:%S UTC'
    client = load_cord_data(project_id, api_key)
    results = read_det_results(result_path)
    upload_results(client, results, root_path, creator_email, gmt_format)
