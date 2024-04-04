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

# generating heatmaps for MSE
data = json.load(open('compression_MSE_Case3.json'))
z_dim = list(data.keys())
epochs = list(data["100"].keys())
m_dim = list(data["100"]["12"].keys())
alphas = list(data["100"]["12"]["480"].keys())
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


# generating plots for iterations

# iter_data = json.load(open('iter_info.json'))
# iter_last = {"10": {"10": 316, "40": 337, "160": 446, "480": 451}, 
#              "50": {"10": 333, "40": 1429, "160": 2000, "480": 1631}, 
#              "75": {"10": 475, "40": 1492, "160": 1057, "480": 1714}, 
#              "100": {"10": 341, "40": 1328, "160": 1438, "480": 1201}, 
#              "150": {"10": 381, "40": 1483, "160": 518, "480": 773}, 
#              "200": {"10": 114, "40": 834, "160": 960, "480": 762}, 
#              "1000": {"10": 79, "40": 316, "160": 553, "480": 583}}
# z_dim = list(iter_last.keys())
# m_dim = list(iter_last["10"].keys())
