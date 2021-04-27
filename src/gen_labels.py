import json
import os.path as osp
import numpy as np
import re
from cord_loader import mkdirs


def gen_gt_information(client, data_root):
    """generate gt.txt file for all data inside the project,
       returns a list of seq_names(if any) which has no labels 
    """
    project = client.get_project()
    obj_type_dict = {}
    feature_hash_dict = {}
    seq_path = data_root + '/train'
    cls_json_path = seq_path
    seq_w_nolabel = []
    #    count = 0
    for label_uid in project.get_labels_list():
        #        if count == 4:
        #            break  # for car dataset only
        #        count += 1
        gt_list = []
        obj_hash_dict = {}
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
        mkdirs(seq_path + f"/{seqs_str}/gt")
        seq_ini_file = osp.join(seq_path, f"{seqs_str}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_path = osp.join(seq_path, f"{seqs_str}", 'gt/gt.txt')
        for frame in list(label['data_units'].values())[0]['labels'].keys():
            for obj in list(label['data_units'].values())[0]['labels'][frame]['objects']:
                if obj['objectHash'] not in obj_hash_dict.keys():
                    obj_hash_dict[obj['objectHash']] = len(obj_hash_dict.keys()) + 1
                if obj['value'] not in obj_type_dict.keys():
                    obj_type_dict[obj['value']] = len(obj_type_dict)
                if obj['featureHash'] not in feature_hash_dict.keys():
                    feature_hash_dict[obj['featureHash']] = len(feature_hash_dict.keys())

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
        try:
            # Prioritize sorting by track id (sort the video frames first, and then sort the track ID)
            idx = np.lexsort(gt_list.T[:2, :])  
        except IndexError:
            seq_w_nolabel.append(seqs_str)
            break
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
        # generate the obj_hash-tid relation json
        tid2objhash = {}
        for key, val in obj_hash_dict.items():
            tid2objhash[val] = key
        with open(osp.join(seq_path, f"{seqs_str}/{seqs_str}_tid2objhash.json"), 'w') as f:
            json.dump(tid2objhash, f, indent=3)

    # Generate cls and id relationship
    cls2id = obj_type_dict
    id2cls = {}
    for key, val in cls2id.items():
        id2cls[val] = key
    with open(osp.join(cls_json_path, 'cls2id.json'), 'w') as f:
        json.dump(cls2id, f, indent=3)
    with open(osp.join(cls_json_path, 'id2cls.json'), 'w') as f:
        json.dump(id2cls, f, indent=3)
    reverse_featureHash_dct = {}
    for key, val in feature_hash_dict.items():
        reverse_featureHash_dct[val] = key
    with open(osp.join(cls_json_path, 'featureHash.json'), 'w') as f:
        json.dump(reverse_featureHash_dct, f, indent=3)

    return seq_w_nolabel


def gen_label_files(seq_names, data_path, save_path, cls2id_dct):
    #    count = 0
    num_of_class = len(cls2id_dct)
    tid_last = dict()
    tid_curr = dict()
    for i in range(num_of_class):
        tid_last[i] = 0
        tid_curr[i] = 0
    for seqs_str in seq_names:
        #        if count == 4:
        #            break  # for car dataset only
        #        count += 1
        ##swap all space in between the seq_name to '_'
        pattern = '(?<=\w)\s(?=\w)'
        try:
            seqs_str = re.sub(pattern, '_', seqs_str)
        except:
            seqs_str = seqs_str
        gt_path = osp.join(data_path, f"train/{seqs_str}")
        gt_file_path = osp.join(gt_path, 'gt', 'gt.txt')
        mkdirs(save_path + f"/train/{seqs_str}/img1")
        seq_ini_file = osp.join(data_path, 'train', f"{seqs_str}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_file = np.loadtxt(gt_file_path, dtype=np.float64, delimiter=',')

        # Prioritize sorting by track id (sort the video frames first, and then sort the track ID)
        idx = np.lexsort(gt_file.T[:2, :])

        gt_file = gt_file[idx, :]

        for idx, (fid, tid, x, y, w, h, mark, cls, vis_ratio) in enumerate(gt_file):
            # frame_id, track_id, top, left, width, height, mark, class, visibility ratio

            # if mark == 0:  # mark为0时忽略(不在当前帧的考虑范围)
            #     continue

            # if vis_ratio <= 0.2:
            #     continue

            fid = int(fid)
            tid = int(tid)

            # Determine whether it is the same track, record the previous track and the current track
            if not tid == tid_last[cls] or idx == 0:  # not has a higher priority than ==
                tid_curr[cls] += 1
                tid_last[cls] = tid

            # center point coordinates of bbox
            x += w / 2
            y += h / 2

            # Write the track id, bbox center point coordinates and width and height to the label (normalized to 0~1)
            # The 0 in the first column is only one category for multi-target detection and tracking by default 
            # (0 is the category)
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                int(cls),
                int(tid_curr[cls]),
                x / seq_width,  # center_x
                y / seq_height,  # center_y
                w / seq_width,  # bbox_w
                h / seq_height,  # bbox_h
            )
            # print(label_str.strip())
            label_fpath = osp.join(
                save_path,
                f"train/{seqs_str}/img1",
                '{:06d}.txt'.format(int(fid)),
            )

            with open(label_fpath, 'a') as f:  # Add the label of each frame in an append
                f.write(label_str)

                
def gen_clsid_info(client, data_root):
    """generate cls2id & id2cls and save as json files
    """
    project = client.get_project()
    obj_type_dict = {}
    feature_hash_dict = {}
    seq_path = data_root + '/train'
    cls_json_path = seq_path
    seq_w_nolabel = []
    #    count = 0
    for label_uid in project.get_labels_list():
        #        if count == 4:
        #            break  # for car dataset only
        #        count += 1
        gt_list = []
        obj_hash_dict = {}
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
        mkdirs(seq_path + f"/{seqs_str}/gt")
        seq_ini_file = osp.join(seq_path, f"{seqs_str}", 'seqinfo.ini')
        seq_info = open(seq_ini_file).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_path = osp.join(seq_path, f"{seqs_str}", 'gt/gt.txt')
        for frame in list(label['data_units'].values())[0]['labels'].keys():
            for obj in list(label['data_units'].values())[0]['labels'][frame]['objects']:
                if obj['objectHash'] not in obj_hash_dict.keys():
                    obj_hash_dict[obj['objectHash']] = len(obj_hash_dict.keys()) + 1
                if obj['value'] not in obj_type_dict.keys():
                    obj_type_dict[obj['value']] = len(obj_type_dict)
                if obj['featureHash'] not in feature_hash_dict.keys():
                    feature_hash_dict[obj['featureHash']] = len(feature_hash_dict.keys())
                pass


    # Generate cls and id relationship
    cls2id = obj_type_dict
    id2cls = {}
    for key, val in cls2id.items():
        id2cls[val] = key
    with open(osp.join(cls_json_path, 'cls2id.json'), 'w') as f:
        json.dump(cls2id, f, indent=3)
    with open(osp.join(cls_json_path, 'id2cls.json'), 'w') as f:
        json.dump(id2cls, f, indent=3)
    reverse_featureHash_dct = {}
    for key, val in feature_hash_dict.items():
        reverse_featureHash_dct[val] = key
    with open(osp.join(cls_json_path, 'featureHash.json'), 'w') as f:
        json.dump(reverse_featureHash_dct, f, indent=3)


# if __name__ == '__main__':
#     project_id = opt.project
#     api_key = opt.api
#     obj_type_dict = {
#         'bus': 4,
#         'car': 0,
#         'motorbike': 3,
#         'pedestrian': 2,
#         'truck': 1,
#     } # 从json读？
#     data_root = '/content/drive/MyDrive/car_data_MCMOT/images/'
#     label_path = '/content/drive/MyDrive/car_data_MCMOT/labels_with_ids/'
#     client = load_cord_data(project_id, api_key)
#     gen_gt_txt(client, obj_type_dict, data_root)
#     gen_label_files(client, data_root, label_path, obj_type_dict)
