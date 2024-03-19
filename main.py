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

from init import seed, BATCH_SIZE, EPOCH_NUM, lr, REAL_LABEL, FAKE_LABEL, device, Z_DIM, M
from create_generator import create_generator
from loss import loss


cudnn.benchmark = True

if __name__ == '__main__':
    print("Random Seed: ", seed)
    print("Batch Size: ", BATCH_SIZE)
    print("Epoch Number: ", EPOCH_NUM)
    print("Learning Rate: ", lr)
    
    # Data preprocessing
    train_dataloader, test_dataloader = get_data()
    # Split dataloader into train and test

    # Use train_dataloader for training
    generator = create_generator(train_dataloader, load=True) # create the generator instance
    
    noise = torch.randn(1, Z_DIM, 1, 1, device=device) # create random noise
    with torch.no_grad():
        fake = generator(noise).detach().cpu()
    print(vutils.make_grid(fake, padding=2, normalize=True))
    
    
    W = torch.randn(M, 64*64)
    W[W > 0] = 1
    W[W <= 0] = -1
    W = W.to(device)
    # Print elements of dataloader
    for data in test_dataloader:
        data = data[0]
        for j,image in enumerate(data):
            image = image.to(device)
            z = torch.randn(1,Z_DIM,1,1, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=0.1)
            for i in range(300):
                loss_value = loss(image, z, generator, W)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                print(loss_value)
                print(z)
            generated_image = generator(z).detach().cpu()
            
            plt.subplot(5, 2, 1+j*2)
            plt.imshow(np.transpose(vutils.make_grid(image.cpu(), padding=2, normalize=True), (1, 2, 0)))
            plt.axis('off')
            
            plt.subplot(5, 2, 2+2*j)
            plt.imshow(np.transpose(vutils.make_grid(generated_image.cpu(), padding=2, normalize=True), (1, 2, 0)))
            plt.axis('off')
            
            if j == 4:
                break
        plt.show()
        break
            
            

    

    
    
    
    