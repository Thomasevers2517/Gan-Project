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

    # Data preprocessing
    train_dataloader, test_dataloader = get_data()
    # Split dataloader into train and test

    # Use train_dataloader for training
    generator = create_generator(train_dataloader, load=False) # create the generator instance
    
    noise = torch.randn(1, Z_DIM, 1, 1, device=device) # create random noise
    with torch.no_grad():
        fake = generator(noise).detach().cpu()
    
    
    W = torch.randn(M, 64*64)
    W[W > 0] = 1
    W[W <= 0] = -1
    W = W.to(device)
    
    from functions import get_z_from_image
    
    original_images = []

    import matplotlib.pyplot as plt
    import numpy as np

    alphas = [0.1, 1,2.5, 5]
    MSE = []  # This will be a 2D list: outer list for each alpha, inner list for MSEs of all images under that alpha

    # Loop over alphas first
    for alpha in alphas:
        alpha_MSEs = []  # To hold MSEs for the current alpha across all images
        example_images = []  # To hold original image examples for plotting
        generated_examples = []  # To hold generated image examples for plotting
        
        for data in test_dataloader:
            images, _ = data  # Assuming your dataloader returns a tuple of images and labels
            print("starting batch")
            for j, image in enumerate(images):
                # Convert image to correct device
                image = image.to(device)
                # Perform operations to get the generated image
                z_opt, info = get_z_from_image(device, image, generator, W, Z_DIM, loss, phase_shift=False, alpha=alpha, iterations=1000, lr=0.05, min_delta=0.02, patience=10)
                generated_image = generator(z_opt).detach().cpu()

                # Compute and store the MSE for the current image
                mse = torch.norm(image.cpu() - generated_image).item()
                alpha_MSEs.append(mse)
                print(f"Image {j+1} - MSE: {mse:.2f} - Alpha: {alpha} - Iterations: {info['last_iter']}")
                # Collect examples for plotting
                if len(example_images) < 4:
                    example_images.append(image.cpu())
                    generated_examples.append(generated_image)
                else:
                    break
            
            # Break after the first batch
            break
        
        # Store the MSEs for the current alpha
        MSE.append(alpha_MSEs)
        
        # Plotting the examples for the current alpha
        fig, axs = plt.subplots(4, 2, figsize=(10, 20))
        for idx, (orig, gen) in enumerate(zip(example_images, generated_examples)):
            axs[idx, 0].imshow(np.transpose(vutils.make_grid(orig, padding=2, normalize=True), (1, 2, 0)))
            axs[idx, 0].set_title(f"Original - Alpha {alpha}")
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(np.transpose(vutils.make_grid(gen, padding=2, normalize=True), (1, 2, 0)))
            axs[idx, 1].set_title(f"Generated - Alpha {alpha}\nMSE: {MSE[-1][idx]:.2f}")
            axs[idx, 1].axis('off')
        
        plt.show()

    # MSE now contains the MSE values for each image for each alpha, organized by alpha


        

        
        
        
        