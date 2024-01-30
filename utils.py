import sys
import os
import logging
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)
        
        
def makedir(dir_name):
    if os.path.exists(os.path.join(dir_name, 'ckpt')):
        print('Output directory already exists')    
    else:
        os.makedirs(os.path.join(dir_name, 'ckpt'))
        os.makedirs(os.path.join(dir_name, 'chains'))
        
        
def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    # file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def save_model(dir_name, epoch, model_name, lr_schedule, G, optG):
    save_dict = {
            'epoch': epoch,
            'lr_schedule': lr_schedule.state_dict(),
            'netG': G.state_dict(),
            'optG': optG.state_dict()
        }
    torch.save(save_dict, f'{dir_name}/ckpt/{model_name}.pth')