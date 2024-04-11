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

# data = json.load(open('Compression_losses/compression_MSE_onlyepoch16.json'))
# save_path = f'MSE_heatmap_Case3_onlyepoch16/'                   #Change the path to save the heatmaps
# #os.mkdir(save_path)
# z_dim = list(data.keys())
# epochs = list(data["1000"].keys())
# print(epochs)
# m_dim = list(data["1000"]["16"].keys())
# alphas = list(data["1000"]["16"]["480"].keys())
# print(alphas)
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
# loss_data = json.load(open('Compression_losses/compression_MSE_onlyepoch16.json'))
# iter_data = json.load(open('iter_info.json'))
# z_dim = list(loss_data.keys())
# print(z_dim)
# epochs = list(loss_data["1000"].keys())
# print(epochs)
# m_dim = list(loss_data["1000"]["16"].keys())
# print(m_dim)
# alphas = list(loss_data["1000"]["16"]["480"].keys())
# print(alphas)   
# epoch='16'
# alpha='0.04'
# iter_last={}
# MSE = pd.DataFrame(columns = ['Z', 'M', 'Last_iter', 'MSE'])
# for z in z_dim:
#     iter_last[z]={}
#     for m in m_dim:
#         iters= [iter_data[z][m][i]['last_iter'] for i in range(5)]
#         iter_last[z][m]=np.mean(iters)
#         df= {'Z': z, 'M': m, 'Last_iter': iter_last[z][m], 'MSE': np.mean(loss_data[z][epoch][m][alpha])} 
#         MSE = pd.concat([MSE, pd.DataFrame([df])], ignore_index=True)


# #colour = {'10': 'violet', '50': 'indigo', '75': 'blue', '100': 'green', '150': 'yellow', '200': 'orange', '1000': 'red'}
# colour = {'10': 'gold', '50': 'darkorange', '75': 'orangered', '100': 'red', '150': 'firebrick', '200': 'maroon', '1000': 'black'}
# marker={'10': 'o', '40': '^', '160': 'D', '480': 'P'}
# fig, ax = plt.subplots(1, 1, figsize=(6,6))
# for i in range(len(MSE)):
#     ax.scatter(MSE['Last_iter'][i], MSE["MSE"][i], c=colour[MSE["Z"][i]], marker=marker[MSE["M"][i]], s=100)

# markers1 = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colour.values()]
# firstlegend= ax.legend(markers1, colour.keys(), numpoints=1, title="k", loc='upper left',bbox_to_anchor=(1.01, 1))
# ax.add_artist(firstlegend)
# markers2 = [plt.Line2D([0,0],[0,0],color='black', marker=marker, linestyle='') for marker in marker.values()]
# ax.legend(markers2, marker.keys(), numpoints=1, title="m", loc='lower left', bbox_to_anchor=(1.01, 0))
# plt.xlabel("Iterations till convergence", fontsize=12)
# plt.ylabel("MSE", fontsize=12)
# plt.title("Iterations vs MSE", fontsize=12)
# plt.show()


#compression_MSE_epoch_16_Case1.json
data = json.load(open('Compression_losses/compression_MSE_z_150_epoch_16_Case3.json'))
save_path = f'MSE_heatmap_z_150_epoch_16_Case3/'                   #Change the path to save the heatmaps
#os.mkdir(save_path)
epoch= '15'
z_dim = list(data.keys())
epochs = list(data["150"].keys())
print(epochs)
m_dim = list(data["150"][epoch].keys())
alphas = list(data["150"][epoch]["480"].keys())
print(alphas)
MSE = np.zeros(( len(m_dim), len(alphas)))
MSEvar = np.zeros(( len(m_dim), len(alphas)))

z='150'
for m in m_dim:
    for alpha in alphas:
        err=np.mean(data[z][epoch][m][alpha])
        var=np.std(data[z][epoch][m][alpha])
        MSE[int(m_dim.index(m)), int(alphas.index(alpha))]=err
        MSEvar[int(m_dim.index(m)), int(alphas.index(alpha))]=var
print(MSE.shape)

sns.heatmap(MSE, annot=True, annot_kws={'va':'bottom', 'size': 'x-large'}, fmt=".2f", cmap='YlOrRd', xticklabels=alphas, yticklabels=m_dim, cbar=False)
sns.heatmap(MSE, annot=MSEvar, annot_kws={'va':'top', 'size': 'large'}, fmt=".2f", cmap='YlOrRd', xticklabels=alphas, yticklabels=m_dim, cbar=False)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14) 
plt.xlabel("alpha values", fontsize=14)
plt.ylabel("m values", fontsize=14)
plt.title("MSE Heatmap Case 3 for k= "+z, fontsize=14)
plt.show()
#plt.savefig(save_path+f"MSE_heatmap_alpha_{alphas[i]}.png")

