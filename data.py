import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

def get_data(path, ratio = 0.8, BATCH_SIZE = 128, X_DIM = 64):
    # Data preprocessing
    dataset = dset.MNIST(root=path, download=True,
                    transform=transforms.Compose([
                    transforms.Resize(X_DIM),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ]))

    # Split dataset into training and testing
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Dataloader for training set
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=8)

    # Dataloader for testing set
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=8)
    
    return train_dataloader, test_dataloader
