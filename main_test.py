import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm

import utils.utils_logger as utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from mlflow import log_metric, log_param, log_artifacts
import mlflow

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    #init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
    #                                                                         net_type='optimizerG')
    #opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    #current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'test'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main testing)
    # ----------------------------------------
    '''
    avg_psnr = 0
    avg_mse = 0
    avg_ssim = 0
    for i, test_data in enumerate(tqdm(test_loader)):
        #idx += 1
        # image_name_ext = os.path.basename(test_data['L_path'][0])
        # img_name, ext = os.path.splitext(image_name_ext)

        # img_dir = os.path.join(opt['path']['images'], img_name)
        # util.mkdir(img_dir)
        #print(test_data)
        model.feed_data(test_data)
        model.test()

        results = model.current_visuals()
        #print(results['E'])
        # E_img = util.tensor2uint(visuals['E'])
        # H_img = util.tensor2uint(visuals['H'])

        # -----------------------
        # save estimated image E
        # -----------------------
        # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        # util.imsave(E_img, save_img_path)

        # -----------------------
        # calculate PSNR
        # -----------------------
        # current_psnr = util.calculate_psnr(visuals['E'], visuals['H'], border=border)
        """
        current_psnr = util.calc_psnr(
            sr=results['E'],
            hr=results['H'],
            scale=border,
            rgb_range=255.0
        )
        current_ssim = util.calculate_ssim(
            img1=results['E'],
            img2=results['H'],
            border=border
        )
        current_mse = util.calculate_mse(
            sr=results['E'],
            hr=results['H'],
            border=border,
            rgb_range=255.0
        )"""
        current_ssim = util.calculate_ssim(
            img1=results['E'],
            img2=results['H'],
            border=border
        )
        current_mse, current_psnr = util.calc_mse_psnr(
            sr=results['E'],
            hr=results['H'],
            scale=border,
            rgb_range=255.0
        )
        # logger.info('{:->4d} | {:<4.2f}dB'.format(idx,
        # image_name_ext,
        #                                                     current_psnr))
        if opt['save_results']:
            util.npyssave(results['E'], opt['saving_path'], test_data['filename'])

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_mse += current_mse

    avg_psnr = avg_psnr / i
    avg_ssim = avg_ssim / i
    avg_mse = avg_mse / i

    # testing log
    logger.info(
        'Average PSNR : {}dB\n'
        'Average SSIM : {:<.3f} \n'
        'Avarage MSE : {}'.format(avg_psnr, avg_ssim, avg_mse))

    return {
        'params': opt['netG'],
        'metrics': {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_mse': avg_mse
        }
    }


if __name__ == '__main__':
    logs = main()
    dns = 'mlflow.brindisi.enea.it'
    user = pw = 'mariano'
    mlflow.set_tracking_uri(f"http://{user}:{pw}@{dns}")
    mlflow.set_experiment("4air-experiment")
    mlflow.log_params(logs['params'])
    mlflow.log_metrics(logs['metrics'])
