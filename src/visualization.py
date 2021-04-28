from cord_loader import load_cord_data
import paths
from lib.opts import opts

import os.path as osp
import os
import numpy as np
import json
import re
import uuid
import random
import pickle
from datetime import datetime
from cord.utils import label_utils
from cord.client import CordClient  # ! pip install cord-client-python

# COLORS = [
#    '#D33115',
#    '#1979a9',
#    '#e07b39',
#    '#edb06b',
#    '#69bdd2',
#    '#80391e',
#    '#1c100b',
#    '#ebdab4',
#    '#042f66',
#    '#b97455',
#    '#44bcd8',
# ]

gen_random_hexcolor = lambda: '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def _read_det_results(result_path):
    results = []
    with open(result_path, 'r') as f:
        for line in f.readlines():
            results.append(line.split(','))
    return results


def _gen_new_obj(tid,
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


def _upload_results(opt,
                    client,
                    seqs, 
                    output_roots,
                    root_path='/content/drive/MyDrive/cattle_data/images/train/',
                    creator_email='grouproject851@gmail.com',
                    gmt_format='%a, %d %b %Y %H:%M:%S UTC',
                    restore=False,
                    clean=True):
    """"""

    id2cls_dct_path = osp.join(root_path, 'id2cls.json')
    featureHash_dct_path = osp.join(root_path, 'featureHash.json')
    with open(id2cls_dct_path, 'r') as f:
        id2cls_dict = json.load(f)
    with open(featureHash_dct_path, 'r') as f:
        featureHash_dct = json.load(f)
    id2color_dict = dict()
    for idx, clsid in enumerate(id2cls_dict.keys()):
        hexcolor_code = gen_random_hexcolor()
        while hexcolor_code in id2color_dict.values():
            hexcolor_code = gen_random_hexcolor()
        id2color_dict[clsid] = hexcolor_code
    project = client.get_project()
    index = 0
    for label_uid in project.get_labels_list():
        try:
            label = client.get_label_row(label_uid)
        except:
            continue
        seqs_str = label['data_title']
        ##swap all space in between the seq_name to '_'
        pattern = '(?<=\w)\s(?=\w)'
        try:
            seqs_str = re.sub(pattern, '_', seqs_str)
        except:
            seqs_str = seqs_str
        if seqs_str in seqs:
            if not opt.overwrite:
                continue
            esle:
                pass
        else:
            continue
        result_path = osp.join(output_roots[index], 'results.txt')
        index += 1
        results = _read_det_results(result_path)
        tid2objhash_dct_path = osp.join(
            root_path, seqs_str,
            f"{seqs_str}_tid2objhash.json",
        )
        try:
            with open(tid2objhash_dct_path, 'r') as f:
                tid2objhash_dct = json.load(f)
        except:
            tid2objhash_dct = {}
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
        ####### for restore gt
        if clean:
            list(label['data_units'].values())[0]['labels'] = {}
            list(label['data_units'].values())[0]['labels']['0'] = {'objects': [], 'classifications': []}
        if restore:
            for key in list(label['data_units'].values())[0]['labels'].keys():
                list(label['data_units'].values())[0]['labels'][key] = {'objects': [], 'classifications': []}
        ####### for restore gt
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
            data_dct = _gen_new_obj(
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


def visualization(opt, seqs, output_roots=[]):
    with open(paths.PATHS_OBJ_PATH, 'rb') as f:
        paths_loader = pickle.load(f)

        gmt_format = '%a, %d %b %Y %H:%M:%S UTC'
        project_id = opt.project
        api_key = opt.api
        creator_email = opt.email
        root_path = paths_loader.TRAIN_DATA_PATH
#         result_path = osp.join(output_root, 'results.txt')
        client = load_cord_data(project_id, api_key)
#         results = _read_det_results(result_path)
        _upload_results(opt, client, seqs, output_roots, root_path, creator_email, gmt_format, restore=False, clean=True)
        
def restore_gt(opt, seq, output_root=None):
    with open(paths.PATHS_OBJ_PATH, 'rb') as f:
        paths_loader = pickle.load(f)
        
        gt_path = osp.join(paths_loader.TRAIN_DATA_PATH, seq, 'gt', 'gt.txt')
        gmt_format = '%a, %d %b %Y %H:%M:%S UTC'
        project_id = opt.project
        api_key = opt.api
        creator_email = opt.email
        root_path = paths_loader.TRAIN_DATA_PATH
        creator_email = 'grouproject851@gmail.com'
        client = load_cord_data(project_id, api_key)
        gts = _read_det_results(gt_path)
        _upload_results(client, gts, root_path, creator_email, gmt_format, restore=True)

    if __name__ == '__main__':
        visualization(opt, seq)
