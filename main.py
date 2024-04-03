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

from create_generator import create_generator
from loss import loss
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
from functions import get_z_from_image, compress_images

import matplotlib.pyplot as plt
import numpy as np
import json
#FIXED
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
CUDA = CUDA and torch.cuda.is_available()
if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

#Changed Parameters
Z_DIM_list = [10, 50, 75, 100,150,200, 1000] 

M_list = [10, 40, 160, 480]
ALPHA_list = [0.004, 0.04, 1, 4]






if __name__ == '__main__':

    # Data preprocessing
    train_dataloader, test_dataloader = get_data(path=DATA_PATH, ratio=0.8)
    # Split dataloader into train and test

    # Create all the generators
    for Z_DIM in Z_DIM_list:
        generator = create_generator(train_dataloader, Z_DIM=Z_DIM, MAX_EPOCH_NUM=MAX_EPOCH_NUM, retrain= False) # create the generator instance

    # for i in range(10):
    #     noise = torch.randn(1, Z_DIM, 1, 1, device=device) # create random noise
    #     with torch.no_grad():
    #         fake = generator(noise).detach().cpu()
    #     vutils.save_image(fake, 'fake_images.png', normalize=True)
    #     plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
    #     plt.axis('off')
    #     plt.show()
    """
    compression_MSE = {}
    for Z_DIM in Z_DIM_list:
        compression_MSE[Z_DIM] = {}
        for epoch in range(MAX_EPOCH_NUM):
            if (epoch==1 or epoch%4==0) and (not epoch==0):
                compression_MSE[Z_DIM][epoch] = {}
                generator = create_generator(train_dataloader, Z_DIM=Z_DIM, MAX_EPOCH_NUM=epoch) # create the generator instance
                for M in M_list:
                    compression_MSE[Z_DIM][epoch][M] = {}
                    for alpha in ALPHA_list:
                                print(f"Z_DIM: {Z_DIM} - M: {M} - Alpha: {alpha} - Epoch: {epoch}")
                                MSE, info = compress_images(M, Z_DIM, alpha, generator, test_dataloader, X_DIM, device, num_images=5, show_images=False)
                                compression_MSE[Z_DIM][epoch][M][alpha] = MSE
                
    print(compression_MSE)
    import json
    with open('compression_MSE.json', 'w') as f:
        json.dump(compression_MSE, f)
    """
    # MSE now contains the MSE values for each image for each alpha, organized by alpha
        
    # Extracting iters info
    iter_info = {}
    iter_last= {}
    for Z_DIM in Z_DIM_list:
        epoch = 16
        iter_info[Z_DIM]={}
        iter_last[Z_DIM]={}
        generator = create_generator(train_dataloader, Z_DIM=Z_DIM, MAX_EPOCH_NUM=epoch)
        for M in M_list:
            alpha=0.04
            MSE, info = compress_images(M, Z_DIM, alpha, generator, test_dataloader, X_DIM, device, num_images=5, show_images=False)
            iter_info[Z_DIM][M] = list(info['loss_hist'])
            iter_last[Z_DIM][M] = info['last_iter']
            print(f"Z_DIM: {Z_DIM} - M: {M} - Alpha: {alpha} - Epoch: {epoch} ---- Iterations: {info['last_iter']}")

    with open('iter_info.json', 'w') as f:
        json.dump(iter_info, f)
    with open('iter_last.json', 'w') as f:
        json.dump(iter_last, f)
