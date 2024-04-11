import torch


def loss(Y, z, generator, W, alpha, phase_shift, device='cuda'):
    """
    Calculates the loss function for find the optimal z.

    Args:
        Y (torch.Tensor): The target tensor.
        z (torch.Tensor): The input tensor to the generator.
        generator (torch.nn.Module): The generator model.
        W (torch.Tensor): The weight matrix.
        alpha (float): The weight for the regularization term.
        phase_shift (bool): Whether to apply phase shift or not.
        device (str, optional): The device to use for calculations. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The calculated loss value.
    """
    
    generated_image = generator(z)
    flat_generated_image = generated_image.view(-1)
    latent_generated = torch.matmul(W, flat_generated_image)
    
    if not phase_shift:
        loss = torch.norm(Y - latent_generated)
    else:
        loss = torch.norm(torch.abs(Y) - torch.abs(latent_generated))
    loss = loss + alpha * torch.norm(z)
        
    return loss