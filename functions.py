import torch

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
