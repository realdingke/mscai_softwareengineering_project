import argparse
import os
import os.path as osp
import random
import re
# pickle
import pickle
import json
import preprocess, gen_labels, gen_data_path, paths, cord_loader, demo, visualization, clean, download_model
from lib.opts import opts
import sys


# def _init_parser():
#     """
#     Parser for command line arguments for the program
#     """
#
#     parser = argparse.ArgumentParser(
#         description="""This helps run the parser functionalities for training a new model
#                     """,
#         allow_abbrev=False,
#     )
#
#     parser.add_argument(
#         "-t",
#         "--test",
#         action="store_true",
#         help="test the whole parser function",
#     )
#
#     parser.add_argument(
#         "--track",
#         action="store_true",
#         help="save the video paths for later tracking",
#     )
#
#     parser.add_argument(
#         "--load_api",
#         action="store_true",
#         help="test the whole parser function",
#     )
#
#     parser.add_argument(
#         "--gen_info",
#         action="store_true",
#         help="print out the root path and basic project info",
#     )
#
#     parser.add_argument(
#         "--project",
#         type=str,
#         nargs='?',
#         default='eec20d90-c014-4cd4-92ea-72341c3a1ab5',
#         help="User input the project ID",
#     )
#
#     parser.add_argument(
#         "--api",
#         type=str,
#         nargs='?',
#         default='T7zAcCv2uvgANe4JhSPDePLMTTf4jN-hYpXu-XdMXaQ',
#         help="User input the API key",
#     )
#
#     #    parser.add_argument(
#     #        "--data_root",
#     #        type=str,
#     #        nargs='?',
#     #        default='/data/',
#     #        help="User input the root directory for storing data",
#     #    )
#
#     parser.add_argument(
#         "-ds",
#         "--dataset_selection",
#         type=str,
#         action="append",
#         default=[],
#         help="User defines the datasets to be used",
#     )
#
#     parser.add_argument(
#         "-vs",
#         "--tracking_video_selection",
#         type=str,
#         action="append",
#         default=[],
#         help="User defines the videos to be directly tracked",
#     )
#
#     parser.add_argument(
#         "--json_name",
#         type=str,
#         nargs='?',
#         default='user_input',
#         help="User input the name for training information json file",
#     )
#
#     parser.add_argument(
#         "-sp",
#         "--split_perc",
#         type=float,
#         nargs='+',
#         action="append",
#         default=[],
#         help="user input split percentage(0-1)",
#     )
#
#     parser.add_argument(
#         "--rseed",
#         type=int,
#         nargs='?',
#         default=10,
#         help="User input the random seed for the splitting of dataset",
#     )
#
#     parser.add_argument(
#         "--rand_split",
#         action="store_true",
#         help="Random split the dataset by specific split_perc"
#     )
#
#     args = parser.parse_args()
#
#     return args


def main(opt):
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments

    result_dict = {}  # for flask return
    if opt.gen_info:
        try:
            import dcn_v2
        except ImportError:
            import pip
            pip.main(['install', '-e',
                      'git+https://github.com/CharlesShang/DCNv2@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2',
                      '--user'])
            sys.path.insert(0, "./src/dcnv2")
            import dcn_v2

        # download pretrained model
        car_model_path = osp.join(paths.ROOT_PATH + '/../exp/mot/car_hrnet_pretrained')
        if not os.path.exists(car_model_path):
            os.makedirs(car_model_path)
        if not osp.exists(car_model_path + '/model_last.pth'):
            download_model.download_file_from_google_drive(
                '1-e6mY2G9PMh3Gvhyis_t6RyNB_JZ03X0',
                car_model_path + '/model_last.pth')

        if not osp.exists(car_model_path + '/opt.txt'):
            download_model.download_file_from_google_drive(
                '1KAn5u6nKRJGhDJBZA_O8hWaaIEh63yXN',
                car_model_path + '/opt.txt')

        cattle_model_path = osp.join(paths.ROOT_PATH + '/../exp/mot/cattle_dla_pretrained')
        if not os.path.exists(cattle_model_path):
            os.makedirs(cattle_model_path)
        if not osp.exists(cattle_model_path + '/model_last.pth'):
            download_model.download_file_from_google_drive(
                '10ekRqMiqY2HYqRsca9TD06LyzF7ik4f8',
                cattle_model_path + '/model_last.pth')

        if not osp.exists(cattle_model_path + '/opt.txt'):
            download_model.download_file_from_google_drive(
                '1-LsPdainXT6au7Nm0LEAJ3_KZyaV9nvH',
                cattle_model_path + '/opt.txt')

        # check pretrained model
        models_name = [direc for direc in os.listdir(paths.MODEL_DIR_PATH)]
        result_dict["models_name"] = models_name

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
        client_data_path = paths.CLIENT_DATA_PATH
        cord_loader.mkdirs(client_data_path)
        # file_name_path = osp.join(
        #     paths.ROOT_PATH,
        #     '..' + paths.DATA_REL_PATH,
        #     'path_names_obj.data',
        # )
        seqs_name_path = osp.join(client_data_path, 'seqs_name_path.data')
        paths_loader.SEQS_NAME_PATH = seqs_name_path

        file_name_path = paths.PATHS_OBJ_PATH
        with open(file_name_path, 'wb') as f:
            pickle.dump(paths_loader, f)

        with open(seqs_name_path, 'wb') as f:
            pickle.dump(seqs_dict, f)
        preprocess.gen_seqini(seqs, data_path)
        gen_labels.gen_clsid_info(client, paths_loader.IMG_ROOT_PATH)
        print(f"The root path is:\n{root_path}")
        # result_dict.update({'root_path': f"The root path is:\n{root_path}"})
        result_dict['root_path'] = root_path
        print('The project contains the below datasets:')
        # result_dict.update({'seq_info': 'The project contains the below datasets:\n'})
        result_dict['seq_info'] = []
        for seq in seqs:
            print(' ' * 6 + seq)
            # result_dict['seq_info'] += (' ' * 6 + seq + '\n')
            result_dict['seq_info'].append(seq)
        print("The videos that have gt labels and can used to train:")
        # result_dict.update({'seq_with_label': "The videos that have gt labels and can used to train:\n"})
        result_dict['seq_with_label'] = []
        for seq in seqs:
            if seq not in empty_seqs:
                print(' ' * 6 + seq)
                result_dict['seq_with_label'].append(seq)
        print("The videos that have no gt labels:")
        # result_dict.update({'seq_without_label': "The videos that have no gt labels:\n"})
        result_dict['seq_without_label'] = []
        for seq in empty_seqs:
            print(' ' * 6 + seq)
            result_dict['seq_without_label'].append(seq)
            # result_dict['seq_without_label'] += (' ' * 6 + seq + '\n')


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

        client_data_path = paths.CLIENT_DATA_PATH
        seqs_name_path = osp.join(client_data_path, 'seqs_name_path.data')
        paths_loader.SEQS_NAME_PATH = seqs_name_path
        # change the data_path to include project_name
        data_path = paths_loader.DATA_PATH

        seqs = cord_loader.gen_seq_name_list(client)
        # user-select datasets to be used
        if len(opt.dataset_selection) != 0:
            seqs = opt.dataset_selection
        else:
            seqs = seqs

        frame_count_dict = preprocess.save_mp4_frame(seqs, data_path)
        result_dict['frame_count'] = frame_count_dict
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
        # justify if the model has been chosen by the user
        if len(opt.specified_model) != 0:
            opt.load_model = osp.join(
                paths.MODEL_DIR_PATH,
                opt.specified_model,
                "model_last.pth")
            opt_path = osp.join(
                paths.MODEL_DIR_PATH,
                opt.specified_model,
                'opt.txt'
            )
            with open(opt_path, "r") as f:
                content = f.read()

            pattern = re.compile('arch: [a-z]+_[0-9]+')
            arch = re.findall(pattern, content)
            opt.arch = arch[0][6:]
        else:
            load_model_ls = opt.load_model.split("/")
            model_name_path = "/".join(load_model_ls[:-1])
            opt_path = osp.join(model_name_path, 'opt.txt')
            with open(opt_path, "r") as f:
                content = f.read()
            pattern = re.compile('arch: [a-z]+_[0-9]+')
            arch = re.findall(pattern, content)
            opt.arch = arch[0][6:]

        file_name_path = paths.PATHS_OBJ_PATH
        with open(file_name_path, 'rb') as f:
            paths_loader = pickle.load(f)
        output_root = None

        # automatically identify reid_cls_ids
        id2cls_path = osp.join(paths_loader.TRAIN_DATA_PATH, 'id2cls.json')
        if os.path.isfile(id2cls_path):
            with open(id2cls_path, 'r') as f:
                data = json.load(f)
            cls_ids_ls = list(data.keys())
            id_str = ", ".join(cls_ids_ls)
            opt.reid_cls_ids = id_str

        if len(opt.input_video) != 0:
            video_name = opt.input_video.split("/")[-1][:-4]
            result = {}
            if opt.output_root == '../results':
                opt.output_root = osp.join(opt.output_root, video_name)
                track_result = demo.run_demo(opt)
                result[video_name] = track_result

        elif len(opt.tracking_video_selection) == 0:
            seqs_name_path = paths_loader.SEQS_NAME_PATH
            with open(seqs_name_path, 'rb') as f:
                seqs_name_dict = pickle.load(f)
            empty_seqs = seqs_name_dict['empty_seqs']
            result = {}
            for seq in empty_seqs:
                empty_seqs_path = osp.join(paths_loader.TRAIN_DATA_PATH, seq, seq)
                if opt.output_root == '../results':
                    opt.output_root = osp.join(opt.output_root, seq)
                    seq_name = seq.split('.')[-2]
                    output_video_path = opt.output_root + f'/{seq_name}_track.mp4'
                    if osp.exists(output_video_path):
                        os.remove(output_video_path)
                    opt.input_video = empty_seqs_path
                    track_result = demo.run_demo(opt)
                    result[seq] = track_result

        else:
            result = {}
            for seq in opt.tracking_video_selection[0]:
                if opt.output_root == '../results':
                    opt.output_root = osp.join(opt.output_root, seq)
                    seq_name = seq.split('.')[-2]
                    output_video_path = opt.output_root + f'/{seq_name}_track.mp4'
                    if osp.exists(output_video_path):
                        os.remove(output_video_path)
                    opt.input_video = osp.join(paths_loader.TRAIN_DATA_PATH, seq, seq)
                    track_result = demo.run_demo(opt)
                    result[seq] = track_result
                    output_root = opt.output_root


        if opt.visual:
            seqs_name_path = paths_loader.SEQS_NAME_PATH
            print(paths_loader.SEQS_NAME_PATH)
            with open(seqs_name_path, 'rb') as f:
                seqs_name_dict = pickle.load(f)
            if seqs_name_dict['empty_seqs'] is None:
                seqs = seqs_name_dict['labeled_seqs']
            else:
                seqs = seqs_name_dict['empty_seqs'] + seqs_name_dict['labeled_seqs']
            for seq in seqs:
                if seq in seqs_name_dict['labeled_seqs']:
                    # usr_input = bool(
                    #     input(f"Warning: Are you sure you want to overwrite the gt of {seq} in Cord? True/False"))
                    if opt.overwrite:
                        visualization.visualization(opt, seq, output_root=output_root)
                else:
                    visualization.visualization(opt, seq, output_root=output_root)
    if opt.restore:
        file_name_path = paths.PATHS_OBJ_PATH
        with open(file_name_path, 'rb') as f:
            paths_loader = pickle.load(f)
        seqs_name_path = paths_loader.SEQS_NAME_PATH
        with open(seqs_name_path, 'rb') as f:
            seqs_name_dict = pickle.load(f)
        seqs = seqs_name_dict['labeled_seqs']
        for seq in seqs:
            visualization.restore_gt(opt, seq)
    if opt.clean:
        clean.clean_files_a()
    if opt.clean_model:
        clean.clean_model()
    return result_dict


if __name__ == "__main__":
    opt = opts().init()
    _ = main(opt)
