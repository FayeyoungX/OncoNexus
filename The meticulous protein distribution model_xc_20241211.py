import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import simps
from scipy.special import softmax
import time
import argparse
import logging
from pathlib import Path
import os

# Configure logging to a file
# logging.basicConfig(filename='test2.log', level=logging.DEBUG, filemode='w',
#                    format='%(asctime)s %(levelname)s: %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')

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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


_shift = 1.
_min_std = 4e-2
_max_std = 0.53231
_min_mu = 0.05
_max_mu = 0.95


def free_ener(mus, sigmas, mass, delta, q, e_field, lb, sat):
    a = sat / mus.shape[0]
    mus = F.sigmoid(mus)
    sigmas = F.sigmoid(sigmas) / 6.  # JUST ASSUME sigma << 1/6
    # entropy: phi_i / m_i Log phi_i
    # all_ent_prot = 1 / mass * (1/4 * ((1 - mus) * 2 * torch.exp(-(1 - mus) ** 2 / 2 / sigmas**2) + mus * torch.exp(-(mus) ** 2 / 2 / sigmas**2) + np.sqrt(2 * np.pi) * sigmas * (-torch.erf(mus/2 ** 0.5/sigmas) + torch.erf((mus-1)/2 ** 0.5/sigmas))) / all_norm - torch.log(all_norm))
    # aq_ent = 0.
    # JUST ASSUME sigma << 1/6
    all_ent_prot = - a / mass / 2 * (
                1 + torch.log(2 * np.pi * sigmas ** 2 / a ** 2))  # make dist broader, same as entropy of water.
    all_ele_prot = - a * e_field * q * mus
    _j_mu = mus[:, None] - sigmas  # mu_i - mu_j
    _j_si = sigmas[:, None] ** 2 + sigmas ** 2  # sigma_i ** 2 + sigma_j ** 2
    _joint_ij = torch.exp(_j_mu ** 2 / 2 / _j_si) / np.sqrt(2 * np.pi) / _j_si ** 0.5
    all_hyd_prot = a * delta - a ** 2 * delta * _joint_ij.sum(dim=1)
    # for i in range(mus.shape[0]):
    #    hyd_prot = a * delta[i]
    #    for j in range(mus.shape[0]):
    #        hyd_prot += -a ** 2 * delta[i] * torch.exp(-(mus[i] - mus[j]) ** 2 / 2 / (sigmas[i] ** 2 + #sigmas[j]**2)) / np.sqrt(2 * np.pi) / torch.sqrt(sigmas[i] ** 2 + sigma[j] ** 2)
    #    all_hyd_prot += hyd_prot
    all_chg_prot = a ** 2 * (_joint_ij * q[:, None] * q).sum(dim=1) * lb
    return 1 / sat * all_ent_prot + all_chg_prot + all_ele_prot + all_hyd_prot


# need to change: layer 1-5
class UNN(nn.Module):
    def __init__(self, feat_phi_size, phi_size=1, hidden_layers=2, hidden_neurons=128):
        super(UNN, self).__init__()
        self.linear1 = nn.Linear(feat_phi_size, hidden_neurons // 2).cuda()
        self.linear2 = nn.Linear(2, hidden_neurons // 2).cuda()
        self.hidden = nn.ModuleList([nn.Linear(hidden_neurons, hidden_neurons).cuda() for _ in range(hidden_layers)])
        self.final = nn.Linear(hidden_neurons, phi_size).cuda()

    def forward(self, data, mu_sig):  # data: n_protein, n_feat; mu_sig: n_protein, 2
        y0 = F.tanh(self.linear1(data))
        y1 = F.tanh(self.linear2(mu_sig))
        y = torch.cat((y0, y1), dim=1)
        for hidden in self.hidden:
            y = torch.tanh(hidden(y))
        return self.final(y)


def gaussian_mean(mus, std):
    return 0.5 * std * (2 * (torch.exp(-mus ** 2 / 2 / std ** 2) - torch.exp(-(mus - 1) ** 2 / 2 / std ** 2)) * std + (
                np.pi * 2) ** 0.5 * (mus * torch.erf(mus / 2 ** 0.5 / std) - torch.sign(mus - 1) * mus * torch.erf(
        torch.abs(mus - 1) / 2 ** 0.5 / std)))


def flory_model(x, feat, mzd, labels, phi0, alm, als, device, e_field=2.0, lb=5.0 / 64., sat=0.2):
    num_protein = feat.shape[0]
    feat = torch.tensor(feat).float().to(device)

    # Define mus and sigmas as leaf tensors
    mus = torch.ones((num_protein, 1), requires_grad=True, device=device)
    sigmas = torch.ones((num_protein, 1), requires_grad=True, device=device)

    u_net = UNN(feat.shape[1]).to(device)
    mse_ = nn.MSELoss()

    # Use mus and sigmas directly as parameters for the optimizer
    optimizer_phi = optim.Adam([mus, sigmas], lr=0.0001, weight_decay=1e-2)
    optimizer_u = optim.Adam(u_net.parameters(), lr=0.0001, weight_decay=1e-4)

    mass = torch.tensor(mzd['MV']).to(device)
    q = torch.tensor(mzd['zi']).to(device)
    delta = torch.tensor(mzd['delta']).to(device)
    delta = delta / mass ** (3. / 2)
    label_set = list(set(labels))
    # print('here'*100)
    # print(label_set)
    nc = len(label_set)
    labels = torch.tensor(labels).to(device)

    def _loss(mus, sigmas, epoch, free_energy=True, label=True, constrain=True):
        loss_f = 0
        loss_c1 = 0
        loss_c2 = 0
        loss_label = 0
        if free_energy:
            loss_f = free_ener(mus, sigmas, mass, delta, q, e_field, lb, sat)  # n_prot

            u_ = u_net(feat, torch.cat([mus, sigmas], dim=1))  # Concatenate mus and sigmas along the last dimension
            loss_f = torch.sum(loss_f) + torch.sum(u_)
            loss_c2 = torch.sum(u_ ** 2)

        if constrain:
            loss_c1 = 0.

        if label:
            mus = F.sigmoid(mus)
            cm = mus
            cm = cm[:, None]
            cm = cm.squeeze(-1)

            loss_label = 0
            for i in range(nc - 1):
                ceni = torch.mean(cm[labels == label_set[i]])
                # print('check'*100)
                # print(labels.reshape(-1).shape)
                # print(label_set)
                # print(loss_label)
                # print(cm)
                loss_label += torch.sum(torch.pdist(cm[labels.reshape(-1) == label_set[i]])) * 2
                for j in range(i + 1, nc):
                    if i < j:
                        cenj = torch.mean(cm[labels == label_set[j]])
                        loss_label += torch.exp(-(torch.abs(ceni - cenj))) * 100
            loss_label /= float(nc) ** 2

        return loss_f, loss_c1, loss_c2, loss_label

    def _train(mus, sigmas, epochs=5000):
        loss_0 = loss_1 = float('inf')
        for epoch in range(epochs):
            s1 = args.s1
            s2 = args.s2
            l1_ = args.l1
            l2_ = 1e4
            l3_ = args.l3
            l4_ = 1

            folder_path = Path('chk') / f'{s1}-{s2}-{l1_}-{l3_}-{args.sigma_delta}-{args.sigma_add}-{args.sigma_rand}'
            folder_path.mkdir(parents=True, exist_ok=True)

            for _ in range(s1):
                loss_f, loss_c1, loss_c2, loss_label = _loss(mus, sigmas, epoch, constrain=False, label=False)
                loss = loss_f + l1_ * loss_c2
                optimizer_phi.zero_grad()
                optimizer_u.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
                optimizer_u.step()
                if loss < loss_1:
                    torch.save(u_net.state_dict(),
                               f"model_unet/{s1}-{s2}-{l1_}-{l3_}-{args.sigma_delta}-{args.sigma_add}-{args.sigma_rand}.pth")
                    loss_1 = loss

            for _ in range(s2):
                loss_f, loss_c1, loss_c2, loss_label = _loss(mus, sigmas, epoch)
                loss = l4_ * loss_f + l2_ * loss_c1 + l3_ * loss_label
                optimizer_u.zero_grad()
                optimizer_phi.zero_grad()
                loss.backward()
                optimizer_phi.step()
                if loss < loss_0:
                    loss_0 = loss
                    torch.save(mus, f"{folder_path}/mus_{epoch:05d}.pth")
                    torch.save(sigmas, f"{folder_path}/sigmas_{epoch:05d}.pth")

            if not epoch % 1:
                torch.save(u_net.state_dict(), f"{folder_path}/u_net_{epoch:05d}.pth")

    _train(mus, sigmas)


def generate_phi(n, n_grid, q, delta, labels, sat, ci=None):
    ci = np.ones(n) / n * sat

    labels = labels.reshape(1, -1)[0]
    pathways = list(set(labels))
    n_pws = len(pathways)
    pw_chg = np.zeros(n_pws)
    pw_idx = {}

    for i in range(n_pws):
        pw_idx[pathways[i]] = i
        pw_chg[i] = q[labels == pathways[i]].sum()
    delta = (delta - delta.min()) / (delta.max() - delta.min())
    mu_ord = np.argsort(pw_chg)

    mu_dict = np.linspace(0.05, 0.95, n_pws)
    mu_dict = np.linspace(0.05, 0.95, q.shape[0])
    mu_ord = np.argsort(q)[::-1]

    x = np.linspace(0, 1, n_grid)
    ret = np.zeros((n + 1, n_grid))
    cw = 1 - ci.sum()
    q1 = ((q - q.min()) / (q.max() - q.min())) * 0.9 + 0.05
    qo = np.argsort(q1)
    m1 = np.linspace(0.05, 0.95, n)
    dq = 0.9 / (q.shape[0] + 1)
    logging.debug(f"{q.max()}: {q1[np.argmax(q)]}, {q.min()}: {q1[np.argmin(q)]}")
    all_mus = []
    all_std = []
    for i in range(n):
        # pw_lb = labels[i]
        # pw_id = pw_idx[pw_lb]
        # mu = q1[i]
        mu = m1[qo[i]]
        # need to change
        sigma2 = -delta[i] * args.sigma_delta + args.sigma_add + args.sigma_rand * (np.random.random() - 0.5)
        # MAKE SURE SIGMA**2 << 1/36
        phi_i = np.exp(-(x - mu + 0.02 * (np.random.random() - 0.5)) ** 2 / (sigma2))
        phi_i = phi_i / simps(phi_i, x) * ci[i]
        ret[i] = phi_i
        all_mus.append((mu-0.5) * 4 - 0.02 * (np.random.random() - 0.5))
        all_std.append(sigma2 ** 0.5)

    ret[:-1] = ret[:-1] / ret[:-1].max()
    ret[:-1] = ret[:-1] / np.sum(ret[:-1], axis=1, keepdims=True)
    ret[-1] = 1 - ret[:-1].sum(axis=0)
    # print(np.array(all_mus))
    # print(F.sigmoid(torch.tensor(np.log(np.array(all_mus)/(1-np.array(all_mus))))))
    return ret[:-1], all_mus, all_std


if __name__ == '__main__':
    logging.basicConfig(filename='test2.log', level=logging.DEBUG, filemode='w',
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    feature_data_path = 'data/psrf.csv'
    feature_data = pd.read_csv(feature_data_path)
    physical_data = feature_data[['gene', 'zi', 'MV']]

    label_data_path = 'data/3nd-label.csv'
    label_data = pd.read_csv(label_data_path)
    label_data = label_data[['Gene', 'Newlable_Cluster']]
    label_data0 = label_data.rename(columns={'Gene': 'gene', 'Newlable_Cluster': 'Labels'})
    # print(label_data0)
    # print(feature_data)
    label_data = pd.merge(label_data0, feature_data, on='gene').iloc[:, :2]
    # print(label_data)
    physical_feature_data = pd.merge(label_data0, feature_data, on='gene')
    mzd = physical_feature_data[['gene', 'zi', 'MV', 'delta']]
    labels = physical_feature_data['Labels'].values
    # print(set(label_data0['gene']) - set(feature_data['gene']), set(feature_data['gene']) - set(label_data0['gene']))
    physical_feature_data.drop(columns=['Labels'], inplace=True)
    # physical_feature_data['gene'] = label_data['gene']
    delta_data = physical_feature_data['delta']
    physical_feature_data.set_index('gene', inplace=True)
    physical_feature_data = physical_feature_data.iloc[:, 1:]
    # print(physical_feature_data)
    physical_feature_data.drop(columns=['total_charge_neutral', 'max_charge_neutral', 'min_charge_neutral'],
                               inplace=True)

    physical_data = pd.merge(label_data0, physical_data, on='gene')
    # physical_data['gene'] = label_data['gene']
    physical_data['delta'] = delta_data

    physical_total1 = physical_feature_data.drop(columns=['zi', 'delta'])
    physical_total = physical_total1.values

    x = np.linspace(0, 1, 200).reshape(-1, 1)

    # mzd = feature_data[['gene', 'zi', 'MV', 'delta']]
    mzd.to_csv('mzd.csv')
    feat = physical_total
    # labels = physical_data['Labels'].values
    lbls = list(set(labels))
    for lbl in lbls:
        logging.info(
            f"Pathway {lbl:3d} Charge\t{mzd['zi'][labels == lbl].sum():.4f}\tDelta\t{mzd['delta'][labels == lbl].sum():.4f}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phi0, alm, als = generate_phi(feat.shape[0], x.shape[0], mzd['zi'], mzd['delta'], labels, 0.1)

    # print(mzd)
    flory_model(x, feat, mzd, labels, phi0, alm, als, device)
