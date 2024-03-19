import torch
from init import device
def loss(image, z, generator, W):
    
    flat_image = image.view(-1)
    flat_image =  flat_image.to(device)
    Y = torch.matmul(W, flat_image)
    
    generated_image = generator(z).to(device)
        
    
    flat_generated_image = generated_image.view(-1)
    latent_generated = torch.matmul(W, flat_generated_image)
    loss = torch.norm(Y - latent_generated)
    return loss