import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from data import get_data
from model import Generator, Discriminator, weights_init

from init import seed, BATCH_SIZE, EPOCH_NUM, lr, REAL_LABEL, FAKE_LABEL, device, Z_DIM
from create_generator import create_generator



cudnn.benchmark = True

if __name__ == '__main__':
    print("Random Seed: ", seed)
    print("Batch Size: ", BATCH_SIZE)
    print("Epoch Number: ", EPOCH_NUM)
    print("Learning Rate: ", lr)
    
    # Data preprocessing
    dataloader = get_data()
    gen = create_generator(dataloader, load=True) # create the generator instance
    
    

    
    
    
    