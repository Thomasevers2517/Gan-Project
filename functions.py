import torch
import torch.nn as nn
from loss import loss
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def get_z_from_image(device, image, generator, W, Z_DIM, loss, phase_shift=True, alpha=None, iterations=300, lr=0.1, patience=10, min_delta=0.001):
    """
    Calculates the latent vector z from an input image using the generator network, with early stopping.

    Args:
        device (torch.device): The device to perform the calculations on.
        image (torch.Tensor): The input image.
        generator (torch.nn.Module): The generator network.
        W (torch.Tensor): The weight matrix for linear transformation.
        Z_DIM (int): The dimension of the latent vector z.
        loss (callable): The loss function to optimize.
        phase_shift (bool, optional): Whether to apply phase shift during optimization. Defaults to True.
        alpha (float, optional): The alpha value for phase shift. Defaults to None.
        iterations (int, optional): The number of optimization iterations. Defaults to 300.
        lr (float, optional): The learning rate for optimization. Defaults to 0.1.
        patience (int, optional): The number of iterations to wait for improvement in loss. Defaults to 10.
        min_delta (float, optional): The minimum change in loss to qualify as an improvement. Defaults to 0.001.

    Returns:
        torch.Tensor: The latent vector z.
        dict: Additional information including the loss history during optimization.
    """
    
    image = image.to(device)
    z = torch.randn(1, Z_DIM, 1, 1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    flat_image = image.view(-1)
    flat_image = flat_image.to(device)
    Y = torch.matmul(W, flat_image)
    loss_hist = []
    
    best_loss = float('inf')
    patience_counter = 0
    last_iter = iterations
    for i in range(iterations):
        loss_value = loss(Y, z, generator, W, phase_shift=phase_shift, alpha=alpha)
        optimizer.zero_grad()
        loss_value.backward()
        loss_hist.append(loss_value.item())
        optimizer.step()
        
        # Check for improvement
        if best_loss - loss_value.item() > min_delta:
            best_loss = loss_value.item()
            patience_counter = 0  # reset patience since we found a better loss
        else:
            patience_counter += 1  # no improvement, so we wait another iteration
        
        if patience_counter >= patience:
            last_iter = i + 1
            # print(f"Early stopping triggered at iteration {i+1}")
            break  # Early stopping condition met
        
    info = {"loss_hist": loss_hist, "last_iter": last_iter}
    return z, info



def compress_images(M, Z_DIM, alpha, generator, test_dataloader, X_DIM, device, epoch=None, show_images=True, num_images=5):
    
    W = torch.randn(M, X_DIM*X_DIM)
    W[W > 0] = 1
    W[W <= 0] = -1
    W = W.to(device)
    

    example_images = []
    generated_examples = []
    MSE = []  # This will be a 2D list: outer list for each alpha, inner list for MSEs of all images under that alpha
    
    # Loop over alphas first
    
    for data in test_dataloader:
        images, _ = data  # Assuming your dataloader returns a tuple of images and labels
        print("starting batch")
        for j, image in enumerate(images):
            # Convert image to correct device
            image = image.to(device)
            # Perform operations to get the generated image
            z_opt, info = get_z_from_image(device, image, generator, W, Z_DIM, loss, phase_shift=True, alpha=alpha, iterations=2000, lr=0.01, min_delta=0.01, patience=10)
            generated_image = generator(z_opt).detach().cpu()

            # Compute and store the MSE for the current image
            mse = torch.norm(image.cpu() - generated_image).item()
            MSE.append(mse)
            print(f"Image {j+1} - MSE: {mse:.2f} - Alpha: {alpha} - Iterations: {info['last_iter']}")
            # Collect examples for plotting
            if len(example_images) < num_images-1:
                example_images.append(image.cpu())
                generated_examples.append(generated_image)
            else:
                break
        # Break after the first batch
        break

    
    
    # Plotting the examples for the current alpha
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    for idx, (orig, gen) in enumerate(zip(example_images, generated_examples)):
        axs[idx, 0].imshow(np.transpose(vutils.make_grid(orig, padding=2, normalize=True), (1, 2, 0)))
        axs[idx, 0].set_title(f"Original - Alpha {alpha}")
        axs[idx, 0].axis('off')

        axs[idx, 1].imshow(np.transpose(vutils.make_grid(gen, padding=2, normalize=True), (1, 2, 0)))
        axs[idx, 1].set_title(f"Generated - Alpha {alpha}\nMSE: {MSE[idx]:.2f}")
        axs[idx, 1].axis('off')
    save_path = f"Compression_Z_{Z_DIM}_M_{M}_alpha_{alpha}_epoch_{epoch}.png"
    plt.savefig(save_path)
    if show_images:
        plt.show()
    else:
        plt.close()
    
    return MSE
    

    
    
    
    