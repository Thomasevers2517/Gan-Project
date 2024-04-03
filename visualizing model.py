#visualizing_model.py
from model import Generator, Discriminator, weights_init  
from torchsummary import summary
import torch.nn as nn
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

Z_DIM = 100
G_HIDDEN = 64
D_HIDDEN = 64
IMAGE_CHANNEL = 1
IMAGE_SIZE = 784

# Create an instance of the Generator and Discriminator
# generator = Generator(Z_DIM, G_HIDDEN, IMAGE_CHANNEL)
# discriminator = Discriminator(D_HIDDEN, IMAGE_CHANNEL)

# # Print the summary of the Generator
# print("Generator Summary:")
# summary(generator, (Z_DIM, 1, 1))

# # Print the summary of the Discriminator
# print("\nDiscriminator Summary:")
# summary(discriminator, (IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

data = json.load(open('compression_MSE.json'))
z_dim = list(data.keys())
epochs = list(data["10"].keys())
m_dim = list(data["10"]["12"].keys())
alphas = list(data["10"]["12"]["480"].keys())
MSE = np.zeros((len(alphas), len(m_dim), len(z_dim)))
MSEvar = np.zeros((len(alphas), len(m_dim), len(z_dim)))
for z in z_dim:
    epoch= '12'
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
    plt.xlabel("Z Dimension")
    plt.ylabel("M Dimension")
    plt.title("MSE Heatmap for Alpha= "+ str(alphas[i]))
    plt.show()

# print(data['1000']['12']['160']['0.004'])
