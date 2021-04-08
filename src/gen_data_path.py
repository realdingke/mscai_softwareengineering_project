import glob
import os
import os.path as osp
import json
import random
# random.seed(10) 


def gen_data_path_half(save_path, 
                       root_path, 
                       seq_list, 
                       mot_path='car_data_MCMOT/images/train'):
    real_path = osp.join(root_path, mot_path)
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


def gen_data_path_val(save_path, 
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = osp.join(root_path, mot_path)
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


def gen_data_path_emb(save_path, 
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = osp.join(root_path, mot_path)
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


def gen_data_path_all(save_path, 
                      root_path, 
                      seq_list, 
                      mot_path='car_data_MCMOT/images/train'):
    real_path = os.path.join(root_path, mot_path)
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
def gen_car_all_path(root_path, 
                     mot_path, 
                     train_name_list=[], 
                     test_name_list=[], 
                     percentage=0.5, 
                     shuffle=True):
    label_root = os.path.join(root_path, "car_data_MCMOT/labels_with_ids/train") #input
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path))] #input
    train_list, test_list = [], []
    for seq_name in seq_names:
        if seq_name in train_name_list:
            seq_path = os.path.join(real_path, seq_name, "img1")
            #label_path = os.path.join(label_root, seq_name, "img1")
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            if shuffle:   
                random.Random(10).shuffle(images)
            len_train = len(images)
            if seq_name in test_name_list:
                len_all = len(images)
                len_train = int(len_all * percentage)
            train_list = sorted(images[:len_train]) # sort train_dataset
            labels = [
                img_path.replace('images', 'labels_with_ids')[:-3] +
                'txt' for img_path in train_list
            ]
            with open('/content/drive/MyDrive/MCMOT/src/data/car_shuffled.train', 'a') as f:
                for i in range(len_train):
                    image = train_list[i]
                    if os.path.isfile(labels[i]):
                        print(image[23:], file=f)                  
            f.close()
        
        if seq_name in test_name_list:
            seq_path = os.path.join(real_path, seq_name, "img1")
            #label_path = os.path.join(label_root, seq_name, "img1")
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            if shuffle:
                random.Random(10).shuffle(images)
            labels = [
                img_path.replace('images', 'labels_with_ids')[:-3] +
                'txt' for img_path in images
            ]
            len_train = 0
            len_all = len(images)
            if seq_name in train_name_list:
                len_train = int(len_all * percentage)
            test_list = sorted(images[len_train:]) # sort test_dataset
            labels = [img_path.replace('images', 'labels_with_ids')[:-3] +
                     'txt' for img_path in test_list]
            with open('/content/drive/MyDrive/MCMOT/src/data/car_shuffled.test', 'a') as f:
                for index, image in enumerate(test_list):
                    if os.path.isfile(labels[index]):
                        print(image[23:], file=f)                  
            f.close()


def generate_json(name, 
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


if __name__ == '__main__':
    root_path = '/content/drive/MyDrive'
    save_path = '/content/drive/MyDrive/MCMOT/src/data/'
    cfg_path = '/content/drive/MyDrive/FairMOT/FairMOT/src/lib/cfg/'
    mot_path = 'car_data_MCMOT/images/train'
    name_lst = ['car_heavy', 'car_all', 'car_high1', 'car_high2']
    seq_list = []
    for name in name_lst:
        if name == 'car_heavy':
            seq_list = ['Heavy traffic.mp4']
        elif name == 'car_high1':
            seq_list = ['Highway traffic.mp4']
        elif name == 'car_high2':
            seq_list = ['Highway traffic2.mp4']
        elif name == 'car_all':
            seq_list = [
                'Light traffic.mp4', 
                'Heavy traffic.mp4', 
                'Highway traffic.mp4', 
                'Highway traffic2.mp4',
            ]
        # gen_data_path_half(save_path, root_path, seq_list, mot_path)
        gen_data_path_val(save_path, root_path, seq_list, mot_path)
        gen_data_path_emb(save_path, root_path, seq_list, mot_path)
        gen_data_path_all(save_path, root_path, seq_list, mot_path)
        generate_json(name, root_path, cfg_path)
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
