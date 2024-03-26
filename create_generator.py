from init import seed, BATCH_SIZE, EPOCH_NUM, lr, REAL_LABEL, FAKE_LABEL, device, Z_DIM, IMAGE_CHANNEL, G_HIDDEN, D_HIDDEN
from model import Generator, Discriminator, weights_init
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
import time
# Directory where plots will be saved
plot_dir = 'training_plots'
os.makedirs(plot_dir, exist_ok=True)

def plot_losses(G_losses, D_losses, epoch, save=True):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig(os.path.join(plot_dir, f'loss_epoch_{epoch}.png'))
    plt.show()
    time.sleep(10)
    plt.close()




def create_generator(dataloader, load=True):
        # Create the generator
    if load:
        netG = Generator()
        netG.load_state_dict(torch.load('netG_weights_D64_G64_Z100.pth'), strict=True)
        return netG.to(device)
    print("Random Seed: ", seed)
    print("Batch Size: ", BATCH_SIZE)
    print("Epoch Number: ", EPOCH_NUM)
    print("Learning Rate: ", lr)
    
    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)
    
        # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(dataloader, 0):
            
            # (1) Update the discriminator with real data
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (3) Update the generator with fake data
            netG.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCH_NUM, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(viz_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # Add this inside your training loop at the end of an epoch
            # For example, after the inner for loop over the dataloader:
            
            # plot_losses(G_losses, D_losses, epoch)   
            iters += 1

        torch.save(netG.state_dict(), f'netG_weights_D{D_HIDDEN}_G{G_HIDDEN}_Z{Z_DIM}.pth')
        print(epoch, G_losses, D_losses)
        
        plot_losses(G_losses, D_losses, EPOCH_NUM-1)
    return netG

