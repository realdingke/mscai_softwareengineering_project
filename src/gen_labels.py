from cord.client import CordClient  # pip install cord-client-python

import json
import os.path as osp
import numpy as np


def mkdirs(d):
    """make dir if not exist"""
    if not osp.exists(d):
        os.makedirs(d)


def load_cord_data(project_id='eec20d90-c014-4cd4-92ea-72341c3a1ab5',
                   api_key='T7zAcCv2uvgANe4JhSPDePLMTTf4jN-hYpXu-XdMXaQ'):
    client = CordClient.initialise(
        project_id,  # Project ID of car
        api_key  # API key
    )

    # Get project info (labels, datasets)
    project = client.get_project()

    return client


def gen_gt_txt(client, obj_type_dict, data_root):
    """generate gt.txt file for car dataset"""
    project = client.get_project()
    seq_path = data_root + 'train/'
    for label_uid in project.get_labels_list():
        gt_list = []
        obj_hash_dict = {}
        label = client.get_label_row(label_uid)
        mkdirs(seq_path + f"{label['data_title']}/gt/")
        seq_ini_file = osp.join(seq_path, f"{label['data_title']}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_path = osp.join(seq_path + f"{label['data_title']}", 'gt/gt.txt')
        for frame in list(label['data_units'].values())[0]['labels'].keys():
            for obj in list(label['data_units'].values())[0]['labels'][frame]['objects']:
                if obj['objectHash'] not in obj_hash_dict.keys():
                    obj_hash_dict[obj['objectHash']] = len(obj_hash_dict.keys()) + 1

                gt_str = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:d},{:d},{:d}\n'.format(
                    int(frame) + 1,
                    obj_hash_dict[obj['objectHash']],
                    obj['boundingBox']['x'] * seq_width,
                    obj['boundingBox']['y'] * seq_height,
                    obj['boundingBox']['w'] * seq_width,
                    obj['boundingBox']['h'] * seq_height,
                    1,
                    obj_type_dict[obj['value']],
                    -1,
                )
                gt_array = np.fromstring(gt_str, dtype=np.float64, sep=',')
                gt_list.append(gt_array)
        gt_list = np.array(gt_list)
        idx = np.lexsort(gt_list.T[:2, :])  # 优先按照track id排序(对视频帧进行排序, 而后对轨迹ID进行排序)
        gt_file = gt_list[idx, :]
        for row in gt_file:
            gt_str = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:d},{:d},{:d}\n'.format(
                int(row[0]),
                int(row[1]),
                row[2],
                row[3],
                row[4],
                row[5],
                int(row[6]),
                int(row[7]),
                int(row[8]),
            )

            with open(gt_path, 'a') as f:
                f.write(gt_str)
        
        # Generate cls and id relationship        
        cls2id = obj_type_dict
        id2cls = {}
        for key,val in cls2id.items():
            id2cls[val] = key
        with open(osp.join(cls_json_path, 'cls2id.json'), 'w') as f:
            json.dump(cls2id, f, indent=3)
        with open(osp.join(cls_json_path, 'id2cls.json'), 'w') as f:
            json.dump(id2cls, f, indent=3)


def gen_label_files(client, data_path, save_path, obj_type_dict):
    project = client.get_project()
    for label_uid in project.get_labels_list():
        label = client.get_label_row(label_uid)
        gt_path = data_path + f"train/{label['data_title']}/"
        gt_file_path = gt_path + 'gt/' + 'gt.txt'
        mkdirs(save_path + f"train/{label['data_title']}/img1/")
        seq_ini_file = osp.join(data_path, 'train', f"{label['data_title']}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_file = np.loadtxt(gt_file_path, dtype=np.float64, delimiter=',')
        
        # 优先按照track id排序(对视频帧进行排序, 而后对轨迹ID进行排序)
        idx = np.lexsort(gt_file.T[:2, :])  
        
        gt_file = gt_file[idx, :]
        num_of_class = len(obj_type_dict) 
        tid_last = dict()
        tid_curr = dict()
        for i in range(num_of_class):
            tid_last[i] = 0
            tid_curr[i] = 0
        for fid, tid, x, y, w, h, mark, cls, vis_ratio in gt_file:
            # frame_id, track_id, top, left, width, height, mark, class, visibility ratio

            # if mark == 0:  # mark为0时忽略(不在当前帧的考虑范围)
            #     continue

            # if vis_ratio <= 0.2:
            #     continue

            fid = int(fid)
            tid = int(tid)

            # 判断是否是同一个track, 记录上一个track和当前track
            if not tid == tid_last[cls]:  # not 的优先级比 == 高
                tid_curr[cls] += 1
                tid_last[cls] = tid

            # bbox中心点坐标
            x += w / 2
            y += h / 2

            # 网label中写入track id, bbox中心点坐标和宽高(归一化到0~1)
            # 第一列的0是默认只对一种类别进行多目标检测跟踪(0是类别)
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                int(cls),
                int(tid_curr[cls]),
                x / seq_width,  # center_x
                y / seq_height,  # center_y
                w / seq_width,  # bbox_w
                h / seq_height, # bbox_h
            )  
            # print(label_str.strip())
            label_fpath = osp.join(
                save_path + f"train/{label['data_title']}/img1/", 
                '{:06d}.txt'.format(int(fid)),
            )

            with open(label_fpath, 'a') as f:  # 以追加的方式添加每一帧的label
                f.write(label_str)


if __name__ == '__main__':
    project_id = 'eec20d90-c014-4cd4-92ea-72341c3a1ab5'
    api_key = 'T7zAcCv2uvgANe4JhSPDePLMTTf4jN-hYpXu-XdMXaQ'
    obj_type_dict = {
        'bus': 4, 
        'car': 0, 
        'motorbike': 3, 
        'pedestrian': 2, 
        'truck': 1,
    } # 从json读？
    data_root = '/content/drive/MyDrive/car_data_MCMOT/images/'
    label_path = '/content/drive/MyDrive/car_data_MCMOT/labels_with_ids/'
    client = load_cord_data(project_id, api_key)
    gen_gt_txt(client, obj_type_dict, data_root)
    gen_label_files(client, data_root, label_path, obj_type_dict)
