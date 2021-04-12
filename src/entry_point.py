import argparse
import os
import os.path as osp
import random
import re
# pickle
import pickle

import preprocess, gen_labels, gen_data_path, paths, cord_loader, demo
from lib.opts import opts


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def _init_parser():
    """
    Parser for command line arguments for the program
    """

    parser = argparse.ArgumentParser(
        description="""This helps run the parser functionalities for training a new model
                    """,
        allow_abbrev=False,
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="test the whole parser function",
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="save the video paths for later tracking",
    )

    parser.add_argument(
        "--load_api",
        action="store_true",
        help="test the whole parser function",
    )

    parser.add_argument(
        "--gen_info",
        action="store_true",
        help="print out the root path and basic project info",
    )

    parser.add_argument(
        "--project",
        type=str,
        nargs='?',
        default='eec20d90-c014-4cd4-92ea-72341c3a1ab5',
        help="User input the project ID",
    )

    parser.add_argument(
        "--api",
        type=str,
        nargs='?',
        default='T7zAcCv2uvgANe4JhSPDePLMTTf4jN-hYpXu-XdMXaQ',
        help="User input the API key",
    )

    #    parser.add_argument(
    #        "--data_root",
    #        type=str,
    #        nargs='?',
    #        default='/data/',
    #        help="User input the root directory for storing data",
    #    )

    parser.add_argument(
        "-ds",
        "--dataset_selection",
        type=str,
        action="append",
        default=[],
        help="User defines the datasets to be used",
    )

    parser.add_argument(
        "-vs",
        "--tracking_video_selection",
        type=str,
        action="append",
        default=[],
        help="User defines the videos to be directly tracked",
    )

    parser.add_argument(
        "--json_name",
        type=str,
        nargs='?',
        default='user_input',
        help="User input the name for training information json file",
    )

    parser.add_argument(
        "-sp",
        "--split_perc",
        type=float,
        nargs='+',
        action="append",
        default=[],
        help="user input split percentage(0-1)",
    )

    parser.add_argument(
        "--rseed",
        type=int,
        nargs='?',
        default=10,
        help="User input the random seed for the splitting of dataset",
    )

    parser.add_argument(
        "--rand_split",
        action="store_true",
        help="Random split the dataset by specific split_perc"
    )

    args = parser.parse_args()

    return args


def main(opt):
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments

    if opt.gen_info:
        project_id = opt.project
        api_key = opt.api
        client = cord_loader.load_cord_data(project_id, api_key)

        pattern = '(?<=\w)\s(?=\w)'
        project_name = client.get_project()['title']
        try:
            project_name = re.sub(pattern, '_', project_name)
        except:
            project_name = project_name

        paths_loader = paths.paths_loader()
        paths_loader.DATA_PATH = osp.join(paths_loader.DATA_PATH, project_name)
        paths_loader.update()

        root_path = paths.ROOT_PATH
        data_path = paths_loader.DATA_PATH
        cord_loader.mkdirs(data_path)
        seqs = cord_loader.gen_seq_name_list(client)

        obj_jsons_list = cord_loader.gen_obj_json(data_path, client=client)
        preprocess.download_mp4(data_path, seqs)
        empty_seqs = cord_loader.judge_video_info(obj_jsons_list)
        seqs_dict = {'labeled_seqs': [seq for seq in seqs if seq not in empty_seqs],
                     'empty_seqs': empty_seqs}
        client_data_path = osp.join(paths_loader.DATA_PATH, '..', 'client_data')
        cord_loader.mkdirs(client_data_path)
        # file_name_path = osp.join(
        #     paths.ROOT_PATH,
        #     '..' + paths.DATA_REL_PATH,
        #     'path_names_obj.data',
        # )
        seqs_name_path = osp.join(client_data_path, 'seqs_name_path.data')
        paths_loader.SEQS_NAME_PATH = seqs_name_path

        file_name_path = osp.join(
            client_data_path,
            'path_names_obj.data',
        )
        with open(file_name_path, 'wb') as f:
            pickle.dump(paths_loader, f)

        with open(seqs_name_path, 'wb') as f:
            pickle.dump(seqs_dict, f)
        print(f"The root path is:\n{root_path}")
        print('The project contains the below datasets:')
        for seq in seqs:
            print(' ' * 6 + seq)
        print("The videos that have gt labels and can used to train:")
        for seq in seqs:
            if seq not in empty_seqs:
                print(' ' * 6 + seq)
        print("The videos that have no gt labels:")
        for seq in empty_seqs:
            print(' ' * 6 + seq)
    if opt.train_track:
        project_id = opt.project
        api_key = opt.api
        client = cord_loader.load_cord_data(project_id, api_key)

        pattern = '(?<=\w)\s(?=\w)'
        project_name = client.get_project()['title']
        try:
            project_name = re.sub(pattern, '_', project_name)
        except:
            project_name = project_name

        paths_loader = paths.paths_loader()
        paths_loader.DATA_PATH = osp.join(paths_loader.DATA_PATH, project_name)
        paths_loader.update()
        root_path = paths.ROOT_PATH

        # change the data_path to include project_name
        data_path = paths_loader.DATA_PATH

        seqs = cord_loader.gen_seq_name_list(client)
        # user-select datasets to be used
        if len(opt.dataset_selection) != 0:
            seqs = opt.dataset_selection
        else:
            seqs = seqs

        preprocess.save_mp4_frame_gen_seqini(seqs, data_path)
        # modified the data root and label path
        data_root = paths_loader.IMG_ROOT_PATH
        print(data_root)
        #        data_root = osp.join(data_path, 'images')
        label_path = paths_loader.LABEL_PATH
        #        label_path = osp.join(data_path, 'labels_with_ids')
        bad_seqs = gen_labels.gen_gt_information(client, data_root)
        seqs = [seq for seq in seqs if seq not in bad_seqs]  # filter out the seqs with no label
        train_data_path = paths_loader.TRAIN_DATA_PATH
        #        train_data_path = osp.join(data_root, 'train')
        cls2id_dct, _ = cord_loader.get_cls_info(train_data_path)
        gen_labels.gen_label_files(seqs, data_root, label_path, cls2id_dct)

        name = opt.json_name
        cfg_path = paths_loader.CFG_DATA_PATH
        json_root_path = paths_loader.DS_JSON_PATH
        gen_data_path.generate_json(name, json_root_path, cfg_path)

        if len(opt.split_perc) == 0:
            opt.split_perc = [[]]
        test_dir_name = gen_data_path.train_test_split(
            root_path=paths_loader.DATA_PATH + '/..',
            project_name=project_name,
            dataset_name_list=seqs,
            percentage=opt.split_perc[0],
            train_file=f"{name}.train",
            test_file=f"{name}.test",
            random_seed=opt.rseed,
            random_split=opt.rand_split
        )
        paths_loader.TEST_DIR_NAME_PATH += test_dir_name
        # save project_name to file_name.data
        client_data_path = osp.join(paths_loader.DATA_PATH, '..', 'client_data')
        file_name_path = osp.join(
            client_data_path,
            'path_names_obj.data',
        )
        with open(file_name_path, 'wb') as f:
            pickle.dump(paths_loader, f)
    if opt.track:
        file_name_path = osp.join(
            paths.ROOT_PATH,
            '..' + paths.DATA_REL_PATH,
            'client_data',
            'path_names_obj.data',
        )
        with open(file_name_path, 'rb') as f:
            paths_loader = pickle.load(f)
        if len(opt.tracking_video_selection) == 0:
            seqs_name_path = paths_loader.SEQS_NAME_PATH
            with open(seqs_name_path, 'rb') as f:
                seqs_name_dict = pickle.load(f)
            empty_seqs = seqs_name_dict['empty_seqs']
            for seq in empty_seqs:
                empty_seqs_path = osp.join(paths_loader.TRAIN_DATA_PATH, seq, seq)
                opt.input_video = empty_seqs_path
                demo.run_demo(opt)
        else:
            for seq in opt.tracking_video_selection[0]:
                opt.input_video = osp.join(paths_loader.TRAIN_DATA_PATH, seq, seq)
                demo.run_demo(opt)


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
