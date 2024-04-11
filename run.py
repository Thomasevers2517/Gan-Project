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
import json

from data import get_data
from model import Generator, Discriminator, weights_init
from create_generator import create_generator
from loss import loss
from functions import get_z_from_image, compress_images

#FIXED PARAMETERS
CUDA = True
DATA_PATH = './data'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
G_HIDDEN = 64
X_DIM =64
D_HIDDEN = 64
MAX_EPOCH_NUM = 16 #16
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

# SET DEVICE
CUDA = CUDA and torch.cuda.is_available()
if CUDA:
    #Make it reproducible
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--n_img', nargs='+', type=int, default=2, help='The number of images to compress (INT)')
parser.add_argument('--k', nargs='+', type=int, default=[150], help='The dimensions of the latent space')
parser.add_argument('--m', nargs='+', type=int, default=[250], help='The number of bits')
parser.add_argument('--alpha', nargs='+', type=float, default=[0.4], help='The alpha values')
parser.add_argument('--epoch', nargs='+', type=int, default=[16], help='The epoch numbers')
parser.add_argument('--noise_std', nargs='+', type=float, default=[0.001], help='The standard deviations of the noise')
parser.add_argument('--case', nargs='+', type=int, default=[3], help='The case numbers')

# Parse the arguments
args = parser.parse_args()

# Now you can use the arguments in your script
num_images = args.n_img
if num_images == 1:
    raise  "Number of images must be greater than 1"
Z_DIM_list = args.k
M_list = args.m
ALPHA_list = args.alpha
EPOCH_list = args.epoch
NOISE_STD_list = args.noise_std
case_list = args.case_list
show_images = True
save_images = True
early_stopping_threshold = 0.02

#Check for invalid parameters
if any(epoch > 16 for epoch in EPOCH_list):
    raise ValueError("Epoch value in EPOCH_list cannot be greater than 15.")

if any(M > 784 for M in M_list):
    raise ValueError("M value in M_list cannot be greater than 784.")

if any(z > 784 for z in Z_DIM_list):
    raise ValueError("Z_DIM value in Z_DIM_list cannot be greater than 784.")




if __name__ == '__main__':

    # Split dataloader into train and test
    train_dataloader, test_dataloader = get_data(path=DATA_PATH, ratio=0.8, num_images=num_images, seed=seed)
    
    print("Data loaded")
    #Run the compression for each case
    for case in case_list:

        if case==1:
            abs_Y=False
            phase_shift_correction = False 
        elif case==2:
            abs_Y=True
            phase_shift_correction = False 
        elif case==3:
            abs_Y=True
            phase_shift_correction = True	    


        compression_MSE = {}
        images =  [image for image in next(iter(test_dataloader))]
        images = images[0]
        
        print(images[0][0][30])

        # Loop over all the parameters
        for Z_DIM in Z_DIM_list:
            compression_MSE[Z_DIM] = {}
            for epoch in EPOCH_list:
                compression_MSE[Z_DIM][epoch] = {}
                
                #create the generator instance, depends on the size of Z_DIM and the epoch. If retrain is false, it will load the existing model if it exists
                generator = create_generator(train_dataloader, Z_DIM=Z_DIM, MAX_EPOCH_NUM=epoch, retrain=False) # create the generator instance
                
                for M in M_list:
                    compression_MSE[Z_DIM][epoch][M] = {}
                    for alpha in ALPHA_list:
                        compression_MSE[Z_DIM][epoch][M][alpha] = {}
                        for noise_std in NOISE_STD_list:
                            
                            #Add noise to the images
                            noisy_images = [image + noise_std*torch.randn_like(image) for image in images]
                            
                            #Compress and decompress the images and get the MSE
                            MSE, info = compress_images(M, Z_DIM, alpha, generator, noisy_images, X_DIM, device, abs_Y, phase_shift_correction,case, num_images=num_images, show_images=show_images, save_images=save_images)
                            
                            compression_MSE[Z_DIM][epoch][M][alpha][noise_std] = MSE
                            print(f"Z_DIM: {Z_DIM} - M: {M} - Alpha: {alpha} - Epoch: {epoch} - Noise STD: {noise_std} ---- MSE: {np.mean(MSE)}")
                            
    #Save the results
        with open(f'Compression_losses/compression_MSE_z_{Z_DIM_list}_m_{M_list}_epoch_{EPOCH_list}_alpha_{ALPHA_list}_noisestd_{NOISE_STD_list}_case{case}.json', 'w') as f:
            json.dump(compression_MSE, f)
    