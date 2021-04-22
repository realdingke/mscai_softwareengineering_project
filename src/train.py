from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import numpy as np
import matplotlib.pyplot as plt
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

# my_visible_devs = '1'  # '0, 3'  # 设置可运行GPU编号
# os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pickle
import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory

# add paths
import paths


# os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64'

def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def encode_image(filename):
    import base64
    e = filename.split(".")[-1]

    img = open(filename,'rb').read()

    data = base64.b64encode(img).decode()

    src = "data:image/{e};base64,{data}".format(e=e, data=data)
    return src

def add_test_loader(opt, data_config, transforms):
    Test_Datast = get_dataset(opt.dataset, opt.task, opt.multi_scale)
    testset_paths = data_config['test']
    testset_root = data_config['root']
    test_dataset = Test_Datast(opt=opt,
                               root=testset_root,
                               paths=testset_paths,
                               img_size=opt.input_wh,
                               augment=False,
                               transforms=transforms)
    opt_2 = opts().update_dataset_info_and_set_heads(opt, test_dataset)
    return test_dataset, opt_2


def plot_loss_curves(opt, data_config, train_losses=None, test_losses=None):
    # path_object = os.path.join(
    #     paths.ROOT_PATH,
    #     '..' + paths.DATA_REL_PATH,
    #     'path_names_obj.data',
    # )
    path_object = paths.PATHS_OBJ_PATH
    with open(path_object, 'rb') as f:
        path_object = pickle.load(f)
    path = os.path.join(path_object.LOSS_CURVES_PATH, opt.exp_id)
    mkdirs(path)
    if len(test_losses) == 0:
        fig, ax = plt.subplots(2, 2)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        ax[0, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['hm']),
                      label='train')
        ax[0, 0].set_ylabel('hm loss')
        ax[0, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['wh']),
                      label='train')
        ax[0, 1].set_ylabel('wh loss')
        ax[1, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['off']),
                      label='train')
        ax[1, 0].set_ylabel('off loss')
        ax[1, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['id']),
                      label='train')
        ax[1, 1].set_ylabel('id loss')
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel('epochs')
                ax[i, j].legend()
        plot_path = os.path.join(path, 'sub_loss.png')
        plt.savefig(plot_path)
        plt.figure()
        plt.plot(np.arange(1, opt.num_epochs + 1),
                 np.array(train_losses['loss']),
                 label='train')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plot_path = os.path.join(path, 'total_loss.png')
        plt.savefig(plot_path)
    else:
        fig, ax = plt.subplots(2, 2)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        ax[0, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['hm']),
                      label='train')
        ax[0, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(test_losses['hm']),
                      label='test')
        ax[0, 0].set_ylabel('hm loss')
        ax[0, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['wh']),
                      label='train')
        ax[0, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(test_losses['wh']),
                      label='test')
        ax[0, 1].set_ylabel('wh loss')
        ax[1, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['off']),
                      label='train')
        ax[1, 0].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(test_losses['off']),
                      label='test')
        ax[1, 0].set_ylabel('off loss')
        ax[1, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(train_losses['id']),
                      label='train')
        ax[1, 1].plot(np.arange(1, opt.num_epochs + 1),
                      np.array(test_losses['id']),
                      label='test')
        ax[1, 1].set_ylabel('id loss')
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel('epochs')
                ax[i, j].legend()

        plot_path = os.path.join(path, 'sub_loss.png')
        plt.savefig(plot_path)
        plt.figure()
        plt.plot(np.arange(1, opt.num_epochs + 1),
                 np.array(train_losses['loss']),
                 label='train')
        plt.plot(np.arange(1, opt.num_epochs + 1),
                 np.array(test_losses['loss']),
                 label='test')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plot_path = os.path.join(path, 'total_loss.png')
        plt.savefig(plot_path)


def save_training_time(opt, data_config, epoch_time=None, total_time=None):
    # path_object = os.path.join(
    #     paths.ROOT_PATH,
    #     '..' + paths.DATA_REL_PATH,
    #     'path_names_obj.data',
    # )
    path_object = paths.PATHS_OBJ_PATH
    with open(path_object, 'rb') as f:
        path_object = pickle.load(f)
    path = path_object.DATA_PATH
    time_path = os.path.join(path, 'training_time.txt')
    epoch_time_str = ", ".join(epoch_time)
    with open(time_path, 'a') as f:
        f.write(f"exp id: {opt.exp_id}, arch: {opt.arch}, epoch: {opt.num_epochs}"
                f", lr: {opt.lr}, batch size:{opt.batch_size}\n")
        f.write(f"total time(min): {total_time}\n")
        f.write(f"epoch time(min): {epoch_time_str}\n")
    f.close()


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task, opt.multi_scale)  # if opt.task==mot -> JointDataset
    f = open(opt.data_cfg)  # choose which dataset to train '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    trainset_paths = data_config['train']  # 训练集路径
    dataset_root = data_config['root']  # 数据集所在目录
    print("Dataset root: %s" % dataset_root)
    f.close()

    # Image data transformations
    transforms = T.Compose([T.ToTensor()])

    # Dataset
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=trainset_paths,
                      img_size=opt.input_wh,
                      augment=True,
                      transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)

    # need to modify
    if opt.add_test_dataset:
        test_dataset, opt_2 = add_test_loader(opt, data_config, transforms)

    logger = Logger(opt)

    # modify
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 多GPU训练
    # print("opt.gpus_str: ", opt.gpus_str)

    # opt.device = torch.device('cuda:0' if opt.gpus[0] >= 0 else 'cpu')  # 设置GPU

    # opt.device = device
    # opt.gpus = my_visible_devs

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    if opt.is_debug:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
    else:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)

    # Get test dataloader
    # if add test dataset then create test loader
    if opt.add_test_dataset:
        if opt.is_debug:
            if opt.multi_scale:
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          drop_last=True)  # debug时不设置线程数(即默认为0)
            else:
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          drop_last=True)  # debug时不设置线程数(即默认为0)
        else:
            if opt.multi_scale:
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          num_workers=opt.num_workers,
                                                          pin_memory=True,
                                                          drop_last=True)
            else:
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          drop_last=True)

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    best = 1e10
    # TODO: add 10 losses lists in order to plot
    if opt.save_time:
        total_time = 0
        epoch_time = []

    if opt.plot_loss:
        train_losses = {'loss': [], 'hm': [], 'wh': [], 'off': [], 'id': []}
        test_losses = {'loss': [], 'hm': [], 'wh': [], 'off': [], 'id': []}
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # Train an epoch
        # TODO: add test loader
        if opt.add_test_dataset:
            log_dict_train, log_dict_test, log_results = trainer.train(epoch,
                                                                       train_loader,
                                                                       test_loader,
                                                                       opt_2)
        else:
            test_loader = None
            opt_2 = None
            log_dict_train, log_results = trainer.train(epoch,
                                                        train_loader,
                                                        test_loader,
                                                        opt_2)

        # append losses to list
        if opt.plot_loss:
            if opt.add_test_dataset:
                train_losses['loss'].append(log_dict_train['loss'])
                train_losses['hm'].append(log_dict_train['hm_loss'])
                train_losses['wh'].append(log_dict_train['wh_loss'])
                train_losses['off'].append(log_dict_train['off_loss'])
                train_losses['id'].append(log_dict_train['id_loss'])

                test_losses['loss'].append(log_dict_test['loss'])
                test_losses['hm'].append(log_dict_test['hm_loss'])
                test_losses['wh'].append(log_dict_test['wh_loss'])
                test_losses['off'].append(log_dict_test['off_loss'])
                test_losses['id'].append(log_dict_test['id_loss'])
            else:
                train_losses['loss'].append(log_dict_train['loss'])
                train_losses['hm'].append(log_dict_train['hm_loss'])
                train_losses['wh'].append(log_dict_train['wh_loss'])
                train_losses['off'].append(log_dict_train['off_loss'])
                train_losses['id'].append(log_dict_train['id_loss'])

        # time
        if opt.save_time:
            epoch_time.append(str(round(log_dict_train['time'], 2)))
            total_time += log_dict_train['time']

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:  # mcmot_last_track or mcmot_last_det
            if opt.id_weight > 0:  # do tracking(detection and re-id)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_track_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
            else:  # only do detection
                # save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                #        epoch, model, optimizer)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 5 == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()

    # plot
    # plot function
    if opt.plot_loss:
        if opt.add_test_dataset:
            plot_loss_curves(opt, data_config, train_losses=train_losses, test_losses=test_losses)
        else:
            plot_loss_curves(opt, data_config, train_losses=train_losses)

    # time function
    if opt.save_time:
        save_training_time(opt, data_config, epoch_time=epoch_time, total_time=total_time)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '0, 1'
    opt = opts().parse()
    # automatically identify reid_cls_ids
    file_name_path = paths.PATHS_OBJ_PATH
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'rb') as f:
            paths_loader = pickle.load(f)
        # automatically identify reid_cls_ids
        id2cls_path = os.path.join(paths_loader.TRAIN_DATA_PATH, 'id2cls.json')
        if os.path.isfile(id2cls_path):
            with open(id2cls_path, 'r') as f:
                data = json.load(f)
            cls_ids_ls = list(data.keys())
            id_str = ", ".join(cls_ids_ls)
            opt.reid_cls_ids = id_str

    run(opt)
