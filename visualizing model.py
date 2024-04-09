#visualizing_model.py
from model import Generator, Discriminator, weights_init  
# from torchsummary import summary
import torch.nn as nn
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Z_DIM = 100
# G_HIDDEN = 64
# D_HIDDEN = 64
# IMAGE_CHANNEL = 1
# IMAGE_SIZE = 784
# Z_DIM = 100
# G_HIDDEN = 64
# D_HIDDEN = 64
# IMAGE_CHANNEL = 1
# IMAGE_SIZE = 784

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

# data = json.load(open('compression_MSE_onlyepoch16.json'))
# save_path = f'MSE_heatmap_Case3_onlyepoch16/'                   #Change the path to save the heatmaps
# #os.mkdir(save_path)
# z_dim = list(data.keys())
# epochs = list(data["1000"].keys())
# print(epochs)
# m_dim = list(data["1000"]["16"].keys())
# alphas = list(data["1000"]["16"]["480"].keys())
# MSE = np.zeros((len(alphas), len(m_dim), len(z_dim)))
# MSEvar = np.zeros((len(alphas), len(m_dim), len(z_dim)))
# for z in z_dim:
#     epoch= '16'
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
#     plt.xlabel("k values")
#     plt.ylabel("m values")
#     plt.title("MSE Heatmap for Alpha= "+ str(alphas[i]))
#     plt.savefig(save_path+f"MSE_heatmap_alpha_{alphas[i]}.png")
#     plt.close()




# generating plots for iterations

# iter_data = json.load(open('iter_info.json'))
# z_dim = list(iter_data.keys())
# print(z_dim)
# m_dim = list(iter_data["10"].keys())
# print(m_dim)
# #images = list(range(iter_data["10"]["40"].shape))
# print(iter_data["10"]["40"][0]['last_iter'])
# iter_last={}
# for z in z_dim:
#     iter_last[z]={}
#     for m in m_dim:
#         losses=np.mean((iter_data[z][m][i]['loss_hist'] for i in range(5)), axis=1)
#         iters= [iter_data[z][m][i]['last_iter'] for i in range(5)]
#         iter_last[z][m]=np.mean(iters)
    #     plt.plot(losses, label='M='+m)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Loss")
    #     plt.title(f"Loss for Z={z}")
    #     plt.legend()
    # plt.show()




# MSE vs Iterations Scatter plot for all Z and M dimensions
loss_data = json.load(open('compression_MSE_onlyepoch16.json'))
iter_data = json.load(open('iter_info.json'))
z_dim = list(loss_data.keys())
print(z_dim)
epochs = list(loss_data["1000"].keys())
print(epochs)
m_dim = list(loss_data["1000"]["16"].keys())
print(m_dim)
alphas = list(loss_data["1000"]["16"]["480"].keys())
print(alphas)   
epoch='16'
alpha='0.04'
iter_last={}
MSE = pd.DataFrame(columns = ['Z', 'M', 'Last_iter', 'MSE'])
for z in z_dim:
    iter_last[z]={}
    for m in m_dim:
        iters= [iter_data[z][m][i]['last_iter'] for i in range(5)]
        iter_last[z][m]=np.mean(iters)
        df= {'Z': z, 'M': m, 'Last_iter': iter_last[z][m], 'MSE': np.mean(loss_data[z][epoch][m][alpha])} 
        MSE = pd.concat([MSE, pd.DataFrame([df])], ignore_index=True)


colour= {'10': 'purple', '50': 'blue', '75': 'skyblue', '100': 'green', '150': 'yellow', '200': 'orange', '1000': 'red'}
marker={'10': 'o', '40': '^', '160': 'D', '480': 'P'}
fig, ax = plt.subplots(1, 1, figsize=(6,6))
for i in range(len(MSE)):
    ax.scatter(MSE['Last_iter'][i], MSE["MSE"][i], c=colour[MSE["Z"][i]], marker=marker[MSE["M"][i]])

markers1 = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colour.values()]
firstlegend= ax.legend(markers1, colour.keys(), numpoints=1, title="k Dimensions", loc='upper left',bbox_to_anchor=(1.01, 1))
ax.add_artist(firstlegend)
markers2 = [plt.Line2D([0,0],[0,0],color='black', marker=marker, linestyle='') for marker in marker.values()]
ax.legend(markers2, marker.keys(), numpoints=1, title="m Dimensions", loc='lower left', bbox_to_anchor=(1.01, 0))
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Iterations vs MSE")
plt.show()