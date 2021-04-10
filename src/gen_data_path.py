import glob
import os
import os.path as osp
import json
import random
import shutil
# random.seed(10) 


def _gen_data_path_half(save_path,
                       root_path, 
                       seq_list, 
                       mot_path='car_data_MCMOT/images/train'):
    real_path = root_path + mot_path
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open(save_path + 'car_all.half', 'w') as f:
        for seq_name in seq_names:
            if seq_name in seq_list:
                seq_path = osp.join(real_path, seq_name, "img1")
                images = sorted(glob.glob(seq_path + '/*.jpg'))  #May be .png
                labels = [
                    img_path.replace('images', 'labels_with_ids')[:-3] + 
                    'txt' for img_path in images
                ]
                # for ~, ~, files in os.walk(label_path):
                # labels = sorted(glob.glob(label_path + '/*.txt'))
                len_all = len(images)
                len_half = int(len_all / 2)
                for i in range(len_half):
                    image = images[i]  # 23 = image.find(mot_path[:5])
                    if osp.isfile(labels[i]):
                        print(image[23:], file=f)  # 23 需要自动化?
    f.close()


def _gen_data_path_val(save_path,
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = root_path + mot_path
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open(save_path + 'car_all.val', 'w') as f:
        for seq_name in seq_names:
            if seq_name in seq_list:
                seq_path = osp.join(real_path, seq_name, "img1")
                images = sorted(glob.glob(seq_path + '/*.jpg'))
                labels = [
                    img_path.replace('images', 'labels_with_ids')[:-3] + 
                    'txt' for img_path in images
                ]
                len_all = len(images)
                len_half = int(len_all / 2)
                for i in range(len_half, len_all):
                    image = images[i]
                    if osp.isfile(labels[i]):
                        print(image[23:], file=f)
    f.close()


def _gen_data_path_emb(save_path,
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = root_path + mot_path
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open(save_path + 'car_all.emb', 'w') as f:
        for seq_name in seq_names:
            if seq_name in seq_list:
                seq_path = osp.join(real_path, seq_name, "img1")
                images = sorted(glob.glob(seq_path + '/*.jpg'))
                labels = [
                    img_path.replace('images', 'labels_with_ids')[:-3] + 
                    'txt' for img_path in images
                ]
                len_all = len(images)
                len_half = int(len_all / 2)
                for i in range(len_half, len_all, 3):
                    image = images[i]
                    if osp.isfile(labels[i]):
                        print(image[23:], file=f)
    f.close()


def _gen_data_path_all(save_path,
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = root_path + mot_path
    seq_names = [s for s in sorted(os.listdir(real_path))]
    with open(save_path + 'car_all.train', 'w') as f:
        for seq_name in seq_names:
            if seq_name in seq_list:
                seq_path = os.path.join(real_path, seq_name, "img1")
                images = sorted(glob.glob(seq_path + '/*.jpg'))
                labels = [
                    img_path.replace('images', 'labels_with_ids')[:-3] + 
                    'txt' for img_path in images
                ]
                # for ~, ~, files in os.walk(label_path):
                # labels = sorted(glob.glob(label_path + '/*.txt'))
                len_all = len(images)
                #   all images
                len_half = int(len_all)
                for i in range(len_half):
                    image = images[i]
                    if os.path.isfile(labels[i]):
                        print(image[23:], file=f)
    f.close()
    

# split dataset
def gen_all_data_path(root_path = '/content/drive/MyDrive' ,
                     project_name = 'car_data_MCMOT', 
                     dataset_name_list=[], 
                     percentage=[], 
                     random_split = False,
                     train_file = 'car_split_all_random.train',
                     test_file = 'car_split_all_random.test',
                     random_seed = 10,
                     test_dir_name = "test_half"):
    """
    split intra video
    """
    data_path = f'{project_name}/images/train'
    label_root = os.path.join(root_path, f"{project_name}/labels_with_ids/train")
    real_path = os.path.join(root_path, label_root)
    test_path = os.path.join(root_path, f'{project_name}/images/test/{test_dir_name}')
    mkdirs(test_path)
    rand_idx = 0
    for idx, seq_name in enumerate(dataset_name_list):
        seq_path = os.path.join(real_path, seq_name, "img1")
        #label_path = os.path.join(label_root, seq_name, "img1")
        labels = sorted(glob.glob(seq_path + '/*.txt'))
        len_all = len(labels)
        len_train = int(len_all * percentage[idx])
        first_frame_idx = int(labels[0][-10:-4])
        if random_split:
            max_idx = len_all - len_train
            rand_idx = random.Random(random_seed).randrange(0, max_idx)  
            train_list = labels[rand_idx: rand_idx+len_train]
        else:
            train_list = labels[:len_train]
        images_train = [
            label_path.replace('labels_with_ids', 'images')[:-3] +
            'jpg' for label_path in train_list
        ]
        # write training image paths

        with open(f'{root_path}/MCMOT/src/data/{train_file}', 'a') as f:
            for i in range(len_train):
                image = images_train[i]
                print(image[23:], file=f)                  
        f.close()

        # test
        origin_seqinfo_path = os.path.join(root_path, project_name, "images/train", seq_name, "seqinfo.ini")
        if len_train != len_all:
            if rand_idx == 0:
                test_list = labels[len_train:] # sort test_dataset
                images_test = [
                    label_path.replace('labels_with_ids', 'images')[:-3] +
                    'jpg' for label_path in test_list
                ]
                # write testing image paths

                with open(f'{root_path}/MCMOT/src/data/{test_file}', 'a') as f:
                    for image in images_test:
                        print(image[23:], file=f)                  
                f.close()

                # generate new testing images

                test_img_path = os.path.join(test_path, f"test_{seq_name}", 'img1')
                mkdirs(test_img_path)
                seqinfo_path = os.path.join(test_path, f"test_{seq_name}")
                shutil.copy(origin_seqinfo_path, seqinfo_path)
                for img_path in images_test:
                    shutil.copy(img_path, test_img_path)
                gt_org_path = os.path.join(root_path, data_path, seq_name, 'gt/gt.txt')
                gt_new_path = os.path.join(test_path, f"test_{seq_name}", 'gt')
                mkdirs(gt_new_path)
                gt_org_list = []

                # read the original gt file

                with open(gt_org_path, 'r') as f:
                    for line in f:
                        gt_org_list.append(line)
                f.close()

                # extract the frame_id of testing data
                gt_new_frame = []
                for path in images_test:
                    gt_new_frame.append(int(path[-10:-4]))
                trk_id_new = 0
                trk_id_old = 0

                # write new gt file for testing data

                with open(gt_new_path+'/gt.txt', 'w') as f:
                    for idx, line in enumerate(gt_org_list):
                        if int(line.split(',')[0]) in gt_new_frame:
                            trk_id = line.split(",")[1]
                            line = line.split(",")
                            frame = int(line[0])
                            if trk_id !=trk_id_old:
                                trk_id_old = trk_id
                                trk_id_new += 1   
                            line[1] = str(trk_id_new)   
                            line[0] = str(frame - len_train - first_frame_idx +1 )
                            obj_str = ",".join(line)
                            f.write(obj_str)
                        
                f.close()
            else: 
                test_list_1 = labels[0:rand_idx]
                images_test_1 = [
                    label_path.replace('labels_with_ids', 'images')[:-3] +
                    'jpg' for label_path in test_list_1
                ]
                with open(f'{root_path}/MCMOT/src/data/{test_file}', 'a') as f:
                    for image in images_test_1:
                        print(image[23:], file=f)                  
                f.close()

                test_img_path = os.path.join(test_path, f"test_1_{seq_name}", 'img1')
                mkdirs(test_img_path)
                seqinfo_path = os.path.join(test_path, f"test_1_{seq_name}")
                shutil.copy(origin_seqinfo_path, seqinfo_path)
                for img_path in images_test_1:
                    shutil.copy(img_path, test_img_path)
                gt_org_path = os.path.join(root_path, data_path, seq_name, 'gt/gt.txt')
                gt_new_path = os.path.join(test_path, f"test_1_{seq_name}", 'gt')
                mkdirs(gt_new_path)
                gt_org_list = []

                # read the original gt file


                with open(gt_org_path, 'r') as f:
                    for line in f:
                        gt_org_list.append(line)
                f.close()

                # extract the frame_id of testing data
                gt_new_frame = []
                for path in images_test_1:
                    gt_new_frame.append(int(path[-10:-4]))
                trk_id_new = 0
                trk_id_old = 0

                # write new gt file for testing data

                with open(gt_new_path+'/gt.txt', 'w') as f:
                    for idx, line in enumerate(gt_org_list):
                        if int(line.split(',')[0]) in gt_new_frame:
                            trk_id = line.split(",")[1]
                            line = line.split(",")
                            frame = int(line[0])
                            if trk_id !=trk_id_old:
                                trk_id_old = trk_id
                                trk_id_new += 1   
                            line[1] = str(trk_id_new)   
                            line[0] = str(frame  -first_frame_idx +1 )
                            obj_str = ",".join(line)

                            f.write(obj_str)
                    
                f.close()

                if (rand_idx+len_train) != len_all:
                    test_list_2 = labels[rand_idx+len_train:]
                    images_test_2 = [
                        label_path.replace('labels_with_ids', 'images')[:-3] +
                        'jpg' for label_path in test_list_2
                    ]
                    with open(f'{root_path}/MCMOT/src/data/{test_file}', 'a') as f:
                        for image in images_test_2:
                            print(image[23:], file=f)                  
                    f.close()
                    test_img_path = os.path.join(test_path, f"test_2_{seq_name}", 'img1')
                    mkdirs(test_img_path)
                    seqinfo_path = os.path.join(test_path, f"test_2_{seq_name}")
                    shutil.copy(origin_seqinfo_path, seqinfo_path)
                    for img_path in images_test_2:
                        shutil.copy(img_path, test_img_path)
                    gt_org_path = os.path.join(root_path, data_path, seq_name, 'gt/gt.txt')
                    gt_new_path = os.path.join(test_path, f"test_2_{seq_name}", 'gt')
                    mkdirs(gt_new_path)
                    gt_org_list = []

                    # read the original gt file

                    with open(gt_org_path, 'r') as f:
                        for line in f:
                            gt_org_list.append(line)
                    f.close()

                    # extract the frame_id of testing data
                    gt_new_frame = []
                    for path in images_test_2:
                        gt_new_frame.append(int(path[-10:-4]))
                    trk_id_new = 0
                    trk_id_old = 0

                    # write new gt file for testing data

                    with open(gt_new_path+'/gt.txt', 'w') as f:
                        for idx, line in enumerate(gt_org_list):
                            if int(line.split(',')[0]) in gt_new_frame:
                                trk_id = line.split(",")[1]
                                line = line.split(",")
                                frame = int(line[0])
                                if trk_id !=trk_id_old:
                                    trk_id_old = trk_id
                                    trk_id_new += 1   
                                line[1] = str(trk_id_new)   
                                line[0] = str(frame - (rand_idx+len_train) - first_frame_idx + 1)
                                obj_str = ",".join(line)

                                f.write(obj_str)
                        
                    f.close()


def _generate_json(name,
                  root_path, 
                  cfg_path='/content/drive/MyDrive/FairMOT/FairMOT/src/lib/cfg/'):
    car_json = {
        "root": f"{root_path}",
        "train":
            {
                f"{name}": f"./data/{name}.train"
            },
        "test_emb":
            {
                f"{name}": f"./data/{name}.emb"
            },
        "test":
            {
                f"{name}": f"./data/{name}.val"
            }
    }
    with open(cfg_path + f"{name}.json", "w") as f:
        json.dump(car_json, f, indent=4)


def generate_paths(name, root_path, seq_list, mot_path):
    save_path = root_path + '/MCMOT/src/data/'
    cfg_path = root_path + '/MCMOT/src/lib/cfg/'
    _gen_data_path_val(save_path, root_path, seq_list, mot_path)
    _gen_data_path_emb(save_path, root_path, seq_list, mot_path)
    _gen_data_path_all(save_path, root_path, seq_list, mot_path)
    _generate_json(name, root_path, cfg_path)

# if __name__ == '__main__':
#     root_path = '/content/drive/MyDrive/'
#     save_path = '/content/drive/MyDrive/MCMOT/src/data/'
#     cfg_path = '/content/drive/MyDrive/MCMOT/src/lib/cfg/'
#     mot_path = 'car_data_MCMOT/images/train'
#     name_lst = ['car_heavy', 'car_all', 'car_high1', 'car_high2']
#     seq_list = []
#     for name in name_lst:
#         if name == 'car_heavy':
#             seq_list = ['Heavy traffic.mp4']
#         elif name == 'car_high1':
#             seq_list = ['Highway traffic.mp4']
#         elif name == 'car_high2':
#             seq_list = ['Highway traffic2.mp4']
#         elif name == 'car_all':
#             seq_list = [
#                 'Light traffic.mp4',
#                 'Heavy traffic.mp4',
#                 'Highway traffic.mp4',
#                 'Highway traffic2.mp4',
#             ]
#         # gen_data_path_half(save_path, root_path, seq_list, mot_path)
#         gen_data_path_val(save_path, root_path, seq_list, mot_path)
#         gen_data_path_emb(save_path, root_path, seq_list, mot_path)
#         gen_data_path_all(save_path, root_path, seq_list, mot_path)
#         generate_json(name, root_path, cfg_path)
    # name_list = [
    #     'Light traffic.mp4', 
    #     'Heavy traffic.mp4', 
    #     'Highway traffic.mp4', 
    #     'Highway traffic2.mp4',
    # ]
    # train_list = ['Highway_traffic_2.mp4']
    # test_list = ['Highway_traffic_2.mp4']
    # name_list = ['Video_of_cattle_1.mp4']
    # gen_car_all_path(root_path, mot_path, train_list, test_list, 0.8, shuffle=True)