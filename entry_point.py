import argparse
import os
import os.path as osp
import random
import re

from src import preprocess, gen_labels, gen_data_path, paths
from src.cord_loader import load_cord_data, gen_seq_name_list, get_cls_info, gen_obj_json, mkdirs


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
        default = [],
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


def main():
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments
    args = _init_parser()

#    if args.load_api:
#        project_id = args.project
#        api_key = args.api
#        args.client = load_cord_data(project_id, api_key)
    if args.gen_info:
        project_id = args.project
        api_key = args.api
        client = load_cord_data(project_id, api_key)
        root_path = paths.ROOT_PATH
        print(f"The root path is:\n{root_path}")
        data_path = root_path + paths.DATA_REL_PATH
        mkdirs(data_path)
        seqs = gen_seq_name_list(client)
        print('The project contains the below datasets:')
        for seq in seqs:
            print(' '*6 + seq)
        gen_obj_json(data_path, client=client)
    if args.test:
        project_id = args.project
        api_key = args.api
        client = load_cord_data(project_id, api_key)
        
        pattern = '(?<=\w)\s(?=\w)'
        project_name = client.get_project()['title']
        try:
            project_name = re.sub(pattern, '_', project_name)
        except:
            project_name = project_name
        root_path = paths.ROOT_PATH
        data_path = osp.join(osp.join(root_path, '..') + paths.DATA_REL_PATH, project_name)
        mkdirs(data_path)
        
        seqs = gen_seq_name_list(client)
        
        # user-select datasets to be used
        if len(args.dataset_selection)!=0:
            seqs = args.dataset_selection
        else:
            seqs = seqs
        
        gen_obj_json(data_path, client=client)
        
        preprocess.download_mp4(data_path, seqs)
        preprocess.save_mp4_frame_gen_seqini(seqs, data_path)
        # modified the data root and label path
        # data_root = paths.IMG_ROOT_PATH
        data_root = osp.join(data_path, 'images')
        # label_path = paths.LABEL_PATH
        label_path = osp.join(data_path, 'labels_with_ids')
        bad_seqs = gen_labels.gen_gt_information(client, data_root)
        seqs = [seq for seq in seqs if seq not in bad_seqs]  #filter out the seqs with no label
        # train_data_path = paths.TRAIN_DATA_PATH
        train_data_path = osp.join(data_root, 'train')
        cls2id_dct, _ = get_cls_info(train_data_path)
        gen_labels.gen_label_files(seqs, data_root, label_path, cls2id_dct)
        
        name = args.json_name
        cfg_path = paths.CFG_DATA_PATH
        gen_data_path.generate_json(name, root_path, cfg_path)

        # modified gen_all_data_path
        # gen_data_path.train_test_split(
        #     root_path = paths.ROOT_PATH,
        #     project_name = paths.DATA_REL_PATH,
        #     dataset_name_list = seqs,
        #     percentage = args.split_perc,
        #     train_file = f"{name}.train",
        #     test_file = f"{name}.test",
        #     random_seed = args.rseed,
        #     test_dir_name = test_data_path,
        #     random_split=args.rand_split,
        # )
        gen_data_path.train_test_split(
            root_path = paths.DATA_PATH,
            project_name = project_name,
            dataset_name_list = seqs,
            percentage = args.split_perc,
            train_file = f"{name}.train",
            test_file = f"{name}.test",
            random_seed = args.rseed,
            random_split = args.rand_split
        )

if __name__ == "__main__":
    main()
