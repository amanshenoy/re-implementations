from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch.nn.functional as f
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import os 
import torchvision

def parser():
    """
    To parse relevant command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--train", dest = 'type', action = 'store_true')
    parser.add_argument('--sample', dest = 'type', action='store_false')
    parser.set_defaults(type=True)
    
    parser.add_argument("--epochs", default = 20, type = int)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--lr", default = 5e-04, type = float)
    parser.add_argument("--save_dir", default = 'saved', type = str)

    parser.add_argument("--num_samples", default = 10, type = int)

    parser.add_argument("--logdir", default = 'runs', type = str)
    parser.add_argument("--run_num", "-r", default = 0, type = int)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)    
    return args

def binarize(image, thresh):
    return (image >= thresh) * 1

def to_one_hot(labels, d, device):
    labels = labels.long().to(device)
    one_hot = torch.FloatTensor(labels.shape[0], d).to(device)
    one_hot.zero_()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

def get_locations_to_append(batch_size, ar_vars, im_shape):
    flattened = torch.arange(ar_vars)
    rewritten = []
    locations = lambda x: [x // im_shape, x % im_shape]

    for ind, element in enumerate(flattened): 
        rewritten.append(locations(element))
    locations = torch.tensor(rewritten).unsqueeze(0).repeat(batch_size, 1, 1)
   
    return locations