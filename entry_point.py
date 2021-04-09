import argparse
import os

from src import preprocess, gen_labels, gen_data_path
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
    # parser.add_argument(
    #     "-c", "--clean", action="store_true", help="cross-validation experiment on clean data set"
    # )
    #
    # parser.add_argument(
    #     "-n", "--noisy", action="store_true", help="cross-validation experiment on noisy data set"
    # )
    #
    # parser.add_argument(
    #     "-p", "--prune", action="store_true", help="perform operation with pruning on the tree"
    # )
    #
    # parser.add_argument(
    #     "-a",
    #     "--all",
    #     action="store_true",
    #     help="run full experiment with cross-validation on "
    #     "clean and noisy data sets then repeat with "
    #     "pruning",
    # )
    #
    # parser.add_argument(
    #     "-sc",
    #     "--showc",
    #     action="store_true",
    #     help="train a single decision tree on the clean data " "set, plotting out the tree",
    # )
    #
    # parser.add_argument(
    #     "-sn",
    #     "--shown",
    #     action="store_true",
    #     help="train a single decision tree on the noisy data " "set, plotting out the tree",
    # )
    #
    # parser.add_argument(
    #     "FOLD", metavar="fold", type=int, help="number of folds to use in cross validation"
    # )

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

    args = parser.parse_args()
    return args


def main():
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments
    args = _init_parser()

    # load data sets
    # clean_data = file_loader.load_data(paths.CLEAN_DATA)
    # noisy_data = file_loader.load_data(paths.NOISY_DATA)

    # if args.all or args.clean:
    #     print("Evaluating trees on the clean dataset")
    #     functions.cross_validate(clean_data, args.FOLD)
    #     print()
    # if args.all or args.noisy:
    #     print("Evaluating trees on the noisy dataset")
    #     functions.cross_validate(noisy_data, args.FOLD)
    #     print()
    # if args.all or (args.prune and args.clean):
    #     print("Evaluating trees on clean data set with pruning")
    #     functions.cross_validate_and_prune(clean_data, args.FOLD)
    #     print()
    # if args.all or (args.prune and args.noisy):
    #     print("Evaluating trees on noisy data set with pruning")
    #     functions.cross_validate_and_prune(noisy_data, args.FOLD)
    # if args.showc:
    #     print(
    #         "Training and Plotting a tree on clean data {} pruning".format(
    #             "with" if args.prune else "without"
    #         )
    #     )
    #     functions.train_and_plot_tree(clean_data, args.FOLD, args.prune)
    # if args.shown:
    #     print(
    #         "Training and Plotting a tree on noisy data {} pruning".format(
    #             "with" if args.prune else "without"
    #         )
    #     )
    #     functions.train_and_plot_tree(noisy_data, args.FOLD, args.prune)
    if args.load_api:
        project_id = args.project
        api_key = args.api
        args.client = load_cord_data(project_id, api_key)
    if args.gen_info:
        root_path = os.getcwd()
        print(root_path)
        data_path = root_path + args.data_root
        mkdirs(data_path)
        seqs = gen_seq_name_list(args.client)
        gen_obj_json(data_path, client=args.client)
    if args.test:
        project_id = args.project
        api_key = args.api
        client = load_cord_data(project_id, api_key)
        
        root_path = os.getcwd()
        data_path = root_path + args.data_root
        mkdirs(data_path)
        
        seqs = gen_seq_name_list(client)
        gen_obj_json(data_path, client=client)
        
        preprocess.download_mp4(data_path, seqs)
        preprocess.save_mp4_frame_gen_seqini(seqs, data_path)
        
        data_root = data_path + 'images/'
        label_path = data_path + 'labels_with_ids/'
        gen_labels.gen_gt_information(client, data_root)
        cls2id_dct, _ = get_cls_info(data_root + 'train/')
        gen_labels.gen_label_files(client, data_root, label_path, cls2id_dct)
        
        name = args.json_name
        mot_path = args.data_root + 'images/train'
        mkdirs(root_path + '/MCMOT/src/data/')
        mkdirs(root_path + '/MCMOT/src/lib/cfg/')
        gen_data_path.generate_paths(name, root_path, seqs, mot_path)


if __name__ == "__main__":
    main()
