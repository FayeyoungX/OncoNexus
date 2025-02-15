import pandas as pd
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os
import re
import torch
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description="Change arguments")
parser.add_argument('s1', type=int)  # 200
parser.add_argument('s2', type=int)  # 150
parser.add_argument('l1', type=float)  # 1
parser.add_argument('l3', type=float)  # 0.6
# parser.add_argument('unet_hidlay', type=int)
parser.add_argument('sigma_delta', type=float)
parser.add_argument('sigma_add', type=float)
parser.add_argument('sigma_rand', type=float)
args = parser.parse_args()

s1 = args.s1
s2 = args.s2
l1_ = args.l1
l3_ = args.l3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# read mus and sigmas
def load_max_epoch_params(folder_path):
    # # List all .pth files in the folder
    # mus_files = [f for f in os.listdir(folder_path) if f.startswith('mus_') and f.endswith('.pth')]
    # sigmas_files = [f for f in os.listdir(folder_path) if f.startswith('sigmas_') and f.endswith('.pth')]
    #
    # # Extract epoch numbers from the filenames using regex
    # epoch_pattern = r'_(\d+)\.pth'  # Assumes epoch is in the form of _epoch.pth, e.g., mus_00005.pth
    #
    # def get_epoch_number(filename):
    #     match = re.search(epoch_pattern, filename)
    #     return int(match.group(1)) if match else -1
    #
    # # Find the mus and sigmas files corresponding to the maximum epoch
    # max_epoch_mus = max(mus_files, key=get_epoch_number)
    # max_epoch_sigmas = max(sigmas_files, key=get_epoch_number)

    max_epoch_mus = 'mus_00125.pth'
    max_epoch_sigmas = 'sigmas_00125.pth'

    # Load the mus and sigmas corresponding to the maximum epoch
    mus = torch.load(os.path.join(folder_path, max_epoch_mus)).to(device)
    sigmas = torch.load(os.path.join(folder_path, max_epoch_sigmas)).to(device)

    mus = F.sigmoid(mus)
    sigmas = F.sigmoid(sigmas) / 6.  # JUST ASSUME sigma << 1/6
    return mus.detach().cpu().numpy().reshape(-1), sigmas.detach().cpu().numpy().reshape(-1)

folder_path = Path('chk') / f'{s1}-{s2}-{l1_}-{l3_}-{args.sigma_delta}-{args.sigma_add}-{args.sigma_rand}'
mu, st = load_max_epoch_params(folder_path)
# mu_sig = torch.load().detach().cpu().numpy() #... Read the saved mus and sigmas as (n_prot, 2)
# mu = mu_sig.T[0]
# st = mu_sig.T[1]

# a 4 gene example:
# mu = np.random.random(4)
# st = np.random.random(4)

# read the labels
label_data_path = 'data/3nd-label.csv'
label_data = pd.read_csv(label_data_path)
label_data = label_data[['Gene', 'Newlable_Cluster']]

# assume pathways are sorted

# a 4 gene example:
# label_data = pd.DataFrame(np.array([['a', 0], ['b', 1], ['c', 2], ['d', 2]]),columns=['Gene', 'Newlabel_Cluster'])

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # colors_for_pathways = [] # number of pathways
# for i in range(label_data.shape[0]):
#     # color for pathway
#     # color = colors_for_pathways[label_data['Newlabel_Cluster'][i]]
#     # alt. color for pathways
#     color = ['red', 'green'][int(label_data['Newlable_Cluster'][i]) % 2]
#     ax.errorbar(i + 1, mu[i], st[i], marker='s', mfc=color, mec=color, ecolor=color, ms=20, ls='', capsize=2)
#
# ax.set_xticks(np.arange(label_data.shape[0]) + 1, labels=label_data['Gene'])
# fig.savefig('draw-test-xc.png', dpi=330)



# Ensure that label_data matches the size of mu and st
num_proteins = len(mu)  # Size of mu (and st)
if label_data.shape[0] != num_proteins:
    print(f"Mismatch: label_data rows ({label_data.shape[0]}) vs mu size ({num_proteins}). Aligning sizes.")
    label_data = label_data.iloc[:num_proteins]  # Trim label_data to match the size of mu and st

# Plot
fig = plt.figure(figsize=(48, 15))
ax = fig.add_subplot(111)

# Assign colors and plot each point with error bars
for i in range(label_data.shape[0]):
    # Alternate colors based on pathway cluster
    color = ['red', 'green'][int(label_data['Newlable_Cluster'].iloc[i]) % 2]
    ax.errorbar(i + 1, mu[i], st[i], marker='s', mfc=color, mec=color, ecolor=color, ms=20, ls='', capsize=2)

# Set x-axis ticks to gene names
ax.set_xticks(np.arange(label_data.shape[0]) + 1)
ax.set_xticklabels(label_data['Gene'], rotation=90)  # Rotate for better readability

# Save the figure
fig.savefig(f'res-variousFig/{args.s1}-{args.s2}-{args.l1}-{args.l3}-{args.sigma_delta}-{args.sigma_add}-{args.sigma_rand}.png', dpi=330)



