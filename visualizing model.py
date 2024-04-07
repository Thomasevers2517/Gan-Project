#visualizing_model.py
from model import Generator, Discriminator, weights_init  
from torchsummary import summary
import torch.nn as nn
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

# # generating heatmaps for MSE
# data = json.load(open('compression_MSE_Case3.json'))
# save_path = f'MSE_heatmap_Case3/'                   #Change the path to save the heatmaps
# #os.mkdir(save_path)

# z_dim = list(data.keys())
# epochs = list(data["1000"].keys())
# print(epochs)
# m_dim = list(data["1000"]["12"].keys())
# alphas = list(data["1000"]["12"]["480"].keys())
# MSE = np.zeros((len(alphas), len(m_dim), len(z_dim)))
# MSEvar = np.zeros((len(alphas), len(m_dim), len(z_dim)))
# for z in z_dim:
#     epoch= '12'
#     for m in m_dim:
#         for alpha in alphas:
#             err=np.mean(data[z][epoch][m][alpha])
#             var=np.std(data[z][epoch][m][alpha])
#             MSE[int(alphas.index(alpha)), int(m_dim.index(m)), int(z_dim.index(z))]=err
#             MSEvar[int(alphas.index(alpha)), int(m_dim.index(m)), int(z_dim.index(z))]=var

# print(MSE.shape)
# for i in range(len(alphas)):
#     sns.heatmap(MSE[i], annot=True, annot_kws={'va':'bottom'}, fmt=".2f", cmap='YlOrRd', xticklabels=z_dim, yticklabels=m_dim)
#     sns.heatmap(MSE[i], annot=MSEvar[i], annot_kws={'va':'top', 'size': 'x-small'}, fmt=".2f", cmap='YlOrRd', xticklabels=z_dim, yticklabels=m_dim, cbar=False)
#     plt.xlabel("Z Dimension")
#     plt.ylabel("M Dimension")
#     plt.title("MSE Heatmap for Alpha= "+ str(alphas[i]))
#     plt.savefig(save_path+f"MSE_heatmap_alpha_{alphas[i]}.png")
#     plt.close()


# generating plots for iterations

iter_data = json.load(open('iter_info.json'))
iter_last = json.load(open('iter_last.json'))
z_dim = list(iter_data.keys())
print(iter_data['1000']['10'][-1])
m_dim = list(iter_data["10"].keys())
print(m_dim)
# plt.plot(iter_data['1000']['480'], label='M=480')
# plt.show()
iter_loss={}
for z in z_dim:
    iter_loss[z]={}
    for m in m_dim:
        iter_loss[z][m]=iter_data[z][m][-1]
        plt.plot(iter_data[z][m], label='M='+m)
        #plt.axvline(x=iter_last[z][m], color='r', linestyle='--', label='Early Stopping')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Loss for Z={z}")
        plt.legend()
    plt.show()

# mse = pd.DataFrame.from_dict(iter_loss)
# mse = pd.DataFrame(mse, index=m_dim)
# mse = mse.reset_index() 
# mse = pd.melt(mse, id_vars=['index'], value_vars=z_dim)  # Melt DataFrame
# mse.columns = ['M', 'Z', 'Loss']

# iter= pd.DataFrame.from_dict(iter_last)
# iter = pd.DataFrame(iter, index=m_dim)
# iter = iter.reset_index()
# iter = pd.melt(iter, id_vars=['index'], value_vars=z_dim)
# iter.columns = ['M', 'Z', 'Iter']
# print(mse)
# print(iter) 
# colour= {'10': 'pink', '50': 'blue', '75': 'green', '100': 'orange', '150': 'red', '200': 'skyblue', '1000': 'brown'}
# marker={'10': 'o', '40': '^', '160': 'D', '480': 'P'}
# fig, ax = plt.subplots()
# for i in range(len(mse)):
#     ax.scatter(iter["Iter"][i], mse["Loss"][i], c=colour[mse["Z"][i]], marker=marker[mse["M"][i]])

# markers1 = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colour.values()]
# firstlegend= ax.legend(markers1, colour.keys(), numpoints=1, title="Z Dimension", loc='upper right',bbox_to_anchor=(1.05, 1))
# ax.add_artist(firstlegend)
# markers2 = [plt.Line2D([0,0],[0,0],color='black', marker=marker, linestyle='') for marker in marker.values()]
# ax.legend(markers2, marker.keys(), numpoints=1, title="M Dimension", loc='lower right', bbox_to_anchor=(1.05, 0))
# plt.xlabel("Iterations")
# plt.ylabel("MSE")
# plt.title("Iterations vs MSE")
# plt.show()



