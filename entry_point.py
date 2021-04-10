import argparse
import os
import random

from src import preprocess, gen_labels, gen_data_path, paths
from src.cord_loader import load_cord_data, gen_seq_name_list, get_cls_info, gen_obj_json, mkdirs


def _init_parser():
    """
    Parser for command line arguments for the program
    """

    parser = argparse.ArgumentParser(
        description="Decision tree training on "
                    "clean and noisy data sets. "
                    "With pruning capabilities",
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
        help="test the whole parser function",
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

    parser.add_argument(
        "--data_root",
        type=str,
        nargs='?',
        default='/data/',
        help="User input the root directory for storing data",
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

    args = parser.parse_args()
    return args


def main():
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments
    args = _init_parser()

    if args.load_api:
        project_id = args.project
        api_key = args.api
        args.client = load_cord_data(project_id, api_key)
    if args.gen_info:
        root_path = paths.ROOT_PATH
        print(root_path)
        data_path = root_path + paths.DATA_REL_PATH
        mkdirs(data_path)
        seqs = gen_seq_name_list(args.client)
        gen_obj_json(data_path, client=args.client)
    if args.test:
        project_id = args.project
        api_key = args.api
        client = load_cord_data(project_id, api_key)
        
        root_path = paths.ROOT_PATH
        data_path = root_path + paths.DATA_REL_PATH
        mkdirs(data_path)
        
        seqs = gen_seq_name_list(client)
        gen_obj_json(data_path, client=client)
        
        preprocess.download_mp4(data_path, seqs)
        preprocess.save_mp4_frame_gen_seqini(seqs, data_path)
        
        data_root = paths.IMG_ROOT_PATH
        label_path = paths.LABEL_PATH
        gen_labels.gen_gt_information(client, data_root)
        train_data_path = paths.TRAIN_DATA_PATH
        cls2id_dct, _ = get_cls_info(train_data_path)
        gen_labels.gen_label_files(client, data_root, label_path, cls2id_dct)
        
        name = args.json_name
        cfg_path = paths.CFG_DATA_PATH
        gen_data_path.generate_json(name, root_path, cfg_path)
        
        test_data_path = 'test_'
        for i in range(len(gen_seq_name_list(client)):
            if len(args.split_perc)!=0:
                test_data_path += str(args.split_perc[i]) + '_'
            else:
                break
        
        gen_all_data_path(
            root_path = paths.ROOT_PATH,
            project_name = paths.DATA_REL_PATH,
            dataset_name_list = seqs,
            percentage = [random.randn()]*len(gen_seq_name_list(client)) if len(args.split_perc)==0,
            train_file = f"{name}.train",
            test_file = f"{name}.test",
            random_seed = args.rseed,
            test_dir_name = test_data_path,           
        )


if __name__ == "__main__":
    main()
