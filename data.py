import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

def get_data(path, ratio=0.8, BATCH_SIZE=128, X_DIM=64, num_images=5, seed=1):
    """
    Retrieves and preprocesses the MNIST dataset.

    Args:
        path (str): The root directory where the dataset will be stored.
        ratio (float, optional): The ratio of the dataset to be used for training. Defaults to 0.8.
        BATCH_SIZE (int, optional): The batch size for the training dataloader. Defaults to 128.
        X_DIM (int, optional): The size to which the images will be resized. Defaults to 64.
        num_images (int, optional): The number of images to be included in the testing dataloader. Defaults to 5.
        seed (int, optional): The seed value for reproducibility. Defaults to 1.

    Returns:
        tuple: A tuple containing the training dataloader and testing dataloader.
    """
    
    CUDA = True
    CUDA = CUDA and torch.cuda.is_available()
    if CUDA:
        #Make it reproducible
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
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
                                        shuffle=False, num_workers=12)

    # Dataloader for testing set. Set num_images to the number of images you want to test. Set num_workers to 0 if reproducibility is needed. 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_images,
                                        shuffle=False, num_workers=0)
    
    return train_dataloader, test_dataloader
