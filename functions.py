import torch
import torch.nn as nn
from loss import loss
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import json
import pandas as pd
import seaborn as sns

        

def get_z_from_image(device, image, generator, W, Z_DIM, alpha, abs_Y, phase_shift, iterations=300, lr=0.1, patience=10, min_delta=0.001):
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
    Y = Y + 0.01*torch.randn_like(Y) # add noise to Y
    if abs_Y:
        # Apply absolute value to Y
        Y = torch.abs(Y)
    loss_hist = []
    
    best_loss = float('inf')
    patience_counter = 0
    last_iter = iterations
    for i in range(iterations):
        loss_value = loss(Y, z, generator, W, alpha=alpha, phase_shift=phase_shift)
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



def compress_images(M, Z_DIM, alpha, generator, images, X_DIM, device, abs_Y, phase_shift, case, show_images=True, num_images=5, save_images=False):
    """
    Compresses a list of images using a generator model. Then the images are decompressed and the mean squared error is calculated.

    Args:
        M (int): The number of random projections.
        Z_DIM (int): The dimension of the latent space.
        alpha (float): The compression ratio.
        generator (torch.nn.Module): The generator model.
        images (list): A list of input images.
        X_DIM (int): The dimension of the input images.
        device (torch.device): The device to perform computations on.
        abs_Y (float): The absolute value of the maximum pixel intensity in the input images.
        phase_shift (float): The phase shift value.
        case (str): The case identifier.
        show_images (bool, optional): Whether to display the compressed images. Defaults to True.
        num_images (int, optional): The number of images to display. Defaults to 5.
        save_images (bool, optional): Whether to save the compressed images. Defaults to False.

    Returns:
        tuple: A tuple containing the list of mean squared errors (MSE) for each image and the list of information dictionaries for each image.
    """
    
    W = torch.randn(M, X_DIM*X_DIM)
    W[W > 0] = 1
    W[W <= 0] = -1
    W = W.to(device)
    
    example_images = []
    generated_examples = []
    MSE = []  # This will be a 2D list: outer list for each alpha, inner list for MSEs of all images under that alpha
    infos = []

    for j, image in enumerate(images):
        image = image.to(device)
        # get the latent vector z
        z_opt, info = get_z_from_image(device, image, generator, W, Z_DIM, alpha, abs_Y=abs_Y, phase_shift=phase_shift, iterations=2000, lr=0.01, min_delta=0.01, patience=10)
        generated_image = generator(z_opt).detach().cpu()
        infos.append(info)
        mse = torch.norm(image.cpu() - generated_image).item()
        MSE.append(mse)
        print(f"Image {j+1} Done - MSE: {mse:.2f} - Iterations: {info['last_iter']} - Loss: {info['loss_hist'][-1]}")
        example_images.append(image.cpu())
        generated_examples.append(generated_image)

    if save_images or show_images:
        fig, axs = plt.subplots(num_images, 2, figsize=(10, 20))
        for idx, (orig, gen) in enumerate(zip(example_images, generated_examples)):
            axs[idx, 0].imshow(np.transpose(vutils.make_grid(orig, padding=2, normalize=True), (1, 2, 0)))
            axs[idx, 0].set_title(f"Original ")
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(np.transpose(vutils.make_grid(gen, padding=2, normalize=True), (1, 2, 0)))
            axs[idx, 1].set_title(f"Generated - Alpha {alpha}  MSE: {MSE[idx]:.2f}")
            axs[idx, 1].axis('off')
        save_path = f"Compression_Z_{Z_DIM}_M_{M}_alpha_{alpha}.png"
        plt.savefig(save_path)
    
    if show_images:
        plt.show()
    else:
        plt.close()

    return MSE, infos
    

def gen_scatterplot(loss_filename= 'Compression_losses/compression_MSE_onlyepoch16.json', iter_filename = 'iter_info.json'):
    """
    Generates a scatter plot of Mean Squared Error (MSE) against iterations till convergence.

    Parameters:
    loss_filename (str): The path to the JSON file containing the loss data. Default is 'Compression_losses/compression_MSE_onlyepoch16.json'.
    iter_filename (str): The path to the JSON file containing the iteration data. Default is 'iter_info.json'.

    Returns:
    The scatter plot uses different colors to represent different 'k' values and different markers to represent different 'm' values.
    The 'k' values are represented in the legend in the upper left corner of the plot.
    The 'm' values are represented in the legend in the lower left corner of the plot.
    The x-axis represents the number of iterations till convergence.
    The y-axis represents the Mean Squared Error (MSE).
    The title of the plot indicates the 'alpha' value used.
    """
    loss_data = json.load(open(loss_filename))
    iter_data = json.load(open(iter_filename))
    z_dim = list(loss_data.keys())
    print(z_dim)
    epochs = list(loss_data[z_dim[0]].keys())
    print(epochs)
    m_dim = list(loss_data[z_dim[0]][epochs[0]].keys())
    print(m_dim)
    alphas = list(loss_data[z_dim[0]][epochs[0]][m_dim[0]].keys())
    print(alphas)  
    iter_last={}
    MSE = pd.DataFrame(columns = ['Z', 'M', 'Last_iter', 'MSE'])
    for alpha in alphas:
        for z in z_dim:
            iter_last[z]={}
            for m in m_dim:
                iters= [iter_data[z][m][i]['last_iter'] for i in range(5)]
                iter_last[z][m]=np.mean(iters)
                df= {'Z': z, 'M': m, 'Last_iter': iter_last[z][m], 'MSE': np.mean(loss_data[z][epochs[-1]][m][alpha])} 
                MSE = pd.concat([MSE, pd.DataFrame([df])], ignore_index=True)

        colour = {'10': 'gold', '50': 'darkorange', '75': 'orangered', '100': 'red', '150': 'firebrick', '200': 'maroon', '1000': 'black'}
        marker={'10': 'o', '40': '^', '160': 'D', '480': 'P'}
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        for i in range(len(MSE)):
            ax.scatter(MSE['Last_iter'][i], MSE["MSE"][i], c=colour[MSE["Z"][i]], marker=marker[MSE["M"][i]], s=100)
        markers1 = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colour.values()]
        firstlegend= ax.legend(markers1, colour.keys(), numpoints=1, title="k", loc='upper left',bbox_to_anchor=(1.01, 1))
        ax.add_artist(firstlegend)
        markers2 = [plt.Line2D([0,0],[0,0],color='black', marker=marker, linestyle='') for marker in marker.values()]
        ax.legend(markers2, marker.keys(), numpoints=1, title="m", loc='lower left', bbox_to_anchor=(1.01, 0))
        plt.xlabel("Iterations till convergence", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.title(f"Iterations vs MSE for alpha= {alpha}", fontsize=12)
        plt.show()
        plt.clf()
        plt.close()
        MSE = MSE.iloc[0:0] 
    
    
def gen_heatmaps_m_vs_alpha(filename='Compression_losses/compression_MSE_epoch_16_Case1.json'):
    """
    Generates a heatmap of Mean Squared Error (MSE) against 'm' and 'alpha' values.

    Parameters:
    filename (str): The path to the JSON file containing the data. Default is 'Compression_losses/compression_MSE_epoch_16_Case1.json'.

    Returns:
    Each heatmap uses the 'YlOrRd' color scheme to represent different MSE values.
    The x-axis represents 'alpha' values.
    The y-axis represents 'm' values.
    The title of the heatmap indicates the 'k' value used.
    The MSE values and their variances are annotated on the heatmap.
    """
    data = json.load(open(filename))
    z_dim = list(data.keys())
    print(z_dim)
    epochs = list(data[z_dim[0]].keys())
    print(epochs)
    m_dim = list(data[z_dim[0]][epochs[0]].keys())
    print(m_dim)
    alphas = list(data[z_dim[0]][epochs[0]][m_dim[0]].keys())
    print(alphas)
    MSE = np.zeros(( len(m_dim), len(alphas)))
    MSEvar = np.zeros(( len(m_dim), len(alphas)))

    for z in z_dim:
        for m in m_dim:
            for alpha in alphas:
                err=np.mean(data[z][epochs[-1]][m][alpha])
                var=np.std(data[z][epochs[-1]][m][alpha])
                MSE[int(m_dim.index(m)), int(alphas.index(alpha))]=err
                MSEvar[int(m_dim.index(m)), int(alphas.index(alpha))]=var
        print(MSE.shape)

        sns.heatmap(MSE, annot=True, annot_kws={'va':'bottom', 'size': 'x-large'}, fmt=".2f", cmap='YlOrRd', xticklabels=alphas, yticklabels=m_dim, cbar=False)
        sns.heatmap(MSE, annot=MSEvar, annot_kws={'va':'top', 'size': 'large'}, fmt=".2f", cmap='YlOrRd', xticklabels=alphas, yticklabels=m_dim, cbar=False)
        plt.xticks(fontsize=14)  
        plt.yticks(fontsize=14) 
        plt.xlabel("alpha values", fontsize=14)
        plt.ylabel("m values", fontsize=14)
        plt.title("MSE Heatmap for k= "+z, fontsize=14)
        plt.show()


def gen_heatmaps_k_vs_m(filename='Compression_losses/compression_MSE_Case3.json'):
    """
    Generates a series of heatmaps of Mean Squared Error (MSE) against 'k' and 'm' values for different 'alpha' values.

    Parameters:
    filename (str): The path to the JSON file containing the data. Default is 'Compression_losses/compression_MSE_Case3.json'.

    Returns:
    Each heatmap uses the 'YlOrRd' color scheme to represent different MSE values.
    The x-axis represents 'k' values.
    The y-axis represents 'm' values.
    The title of each heatmap indicates the 'alpha' value used.
    The MSE values and their standard deviations are annotated on each heatmap.
    """
    data = json.load(open(filename))
    z_dim = list(data.keys())
    print(z_dim)
    epochs = list(data[z_dim[0]].keys())
    print(epochs)
    m_dim = list(data[z_dim[0]][epochs[0]].keys())
    print(m_dim)
    alphas = list(data[z_dim[0]][epochs[0]][m_dim[0]].keys())
    print(alphas)
    MSE = np.zeros((len(alphas), len(m_dim), len(z_dim)))
    MSEvar = np.zeros((len(alphas), len(m_dim), len(z_dim)))
    for z in z_dim:
        for epoch in epochs:
            for m in m_dim:
                for alpha in alphas:
                    err=np.mean(data[z][epoch][m][alpha])
                    var=np.std(data[z][epoch][m][alpha])
                    MSE[int(alphas.index(alpha)), int(m_dim.index(m)), int(z_dim.index(z))]=err
                    MSEvar[int(alphas.index(alpha)), int(m_dim.index(m)), int(z_dim.index(z))]=var
    print(MSE.shape)
    for i in range(len(alphas)):
        sns.heatmap(MSE[i], annot=True, annot_kws={'va':'bottom'}, fmt=".2f", cmap='YlOrRd', xticklabels=z_dim, yticklabels=m_dim)
        sns.heatmap(MSE[i], annot=MSEvar[i], annot_kws={'va':'top', 'size': 'x-small'}, fmt=".2f", cmap='YlOrRd', xticklabels=z_dim, yticklabels=m_dim, cbar=False)
        plt.xlabel("k values")
        plt.ylabel("m values")
        plt.title("MSE Heatmap for Alpha= "+ str(alphas[i]))
        plt.show()
        #plt.savefig(save_path+f"MSE_heatmap_alpha_{alphas[i]}.png")
        plt.close()

    
