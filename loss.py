import torch


def loss(Y, z, generator, W, phase_shift = True, alpha = None, device = 'cuda'):
    
    assert not(phase_shift and (alpha is None)) # if phase_shift is True, alpha should not be None
    
    generated_image = generator(z)
    flat_generated_image = generated_image.view(-1)
    latent_generated = torch.matmul(W, flat_generated_image)
    
    if not phase_shift:
        loss = torch.norm(Y - latent_generated)
    else:
        loss = torch.norm(torch.abs(Y) - torch.abs(latent_generated))
        loss = loss +  alpha  * torch.norm(z) # 1e2 * torch.exp(-0.5*z_sq_norm)#0.00001*torch.norm(z)
        
    #1e1 * torch.norm(z) got 3/5
    return loss