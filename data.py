import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from init import DATA_PATH, BATCH_SIZE, X_DIM
def get_data():
    # Data preprocessing
    dataset = dset.MNIST(root=DATA_PATH, download=True,
                    transform=transforms.Compose([
                    transforms.Resize(X_DIM),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ]))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=2)
    return dataloader
