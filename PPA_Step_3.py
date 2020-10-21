"""
PPA Step-3: LASSO regression

Author: RJ Liu (rl58@rice.edu)
Last update: 2020-04-04
"""

import numpy as np
import os
import sys
import fnmatch
import pandas as pd
from scipy.io import savemat
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV

"""
installed all the libraries above
"""

home_dir = './'
ft_file_dir = home_dir + 'data'
trait_dir = home_dir + 'trait'
endpoint_dir = home_dir + 'ending_point'
input_dir = home_dir + 'clustering'
output_dir = home_dir + 'lasso'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
brainet_dir = home_dir + 'brainet'
if not os.path.exists(brainet_dir):
    os.mkdir(brainet_dir)
n_clusters = int(sys.argv[1])
mode = sys.argv[2]

labels = np.loadtxt(input_dir + '/' + 'cluster_label_' + str(n_clusters) + '.txt')
n_tracts = np.loadtxt(endpoint_dir + '/' + 'n_tracts.txt')
cum_tracts = np.zeros(shape=(1+n_tracts.shape[0],))
cum_tracts[1:] = n_tracts
mu0 = np.loadtxt(input_dir + '/' + 'cluster_means_' + str(n_clusters) + '.txt')
mu = mu0[:-1, :]
w = np.loadtxt(input_dir + '/' + 'cluster_w_' + str(n_clusters) + '.txt')
X = w[:, :-1]
trait = np.loadtxt(trait_dir + '/' + mode + '.txt')
y = trait[:, 1]
dat = np.concatenate((X, y.reshape((-1, 1))), axis=1)

start = 400
tol = 1e-4


def mse_lasso(dat_all):
    np.random.shuffle(dat_all)
    reg = LassoCV(cv=5, random_state=0, tol=tol)
    reg.fit(dat_all[:start, :-1], dat_all[:start, -1].reshape((-1,)))
    y_ = reg.predict(dat_all[start:, :-1])
    return mean_squared_error(dat_all[start:, -1], y_)


lasso = LassoCV(cv=5, random_state=0, tol=tol)
lasso.fit(dat[:, :-1], dat[:, -1].reshape((-1,)))
beta = lasso.coef_
mse = np.mean([mse_lasso(dat) for i in range(100)])
mat = {'beta': beta, 'mse': mse}
savemat(output_dir + '/' + 'mse_' + mode + str(n_clusters) + '.mat', mat)

beta_idx = np.where(beta != 0)[0]
cluster_idx = []
for t in beta_idx:
    c_idx_t = np.where(labels == t)[0]
    cluster_idx.append(c_idx_t)
# print(cluster_idx)

# screen all fiber tracking files
sub_id_list = np.loadtxt(home_dir + 'subjects_0.txt')
n = len(sub_id_list)
tract_mat = np.zeros(shape=(3, 1))
length_all = np.zeros(shape=(2, 1))
print('load the fiber tracking file \n')
for k in range(n):
    # print(sub_id_list[k])
    idx_list_tmp = np.intersect1d(np.where(cluster_idx >= cum_tracts[k])[0], np.where(cluster_idx < cum_tracts[k+1])[0])
    idx_list = idx_list_tmp.astype(int)
    cluster_list = cluster_idx[idx_list] - cum_tracts[k]
    file_list = fnmatch.filter(os.listdir(ft_file_dir), str(int(sub_id_list[k])) + '*.txt')
    file_name = file_list[0]
    file_path = ft_file_dir + '/' + file_name
    dat = pd.read_csv(file_path, header=None)
    for j in cluster_list:
        tmp0 = dat.loc[j][0]
        tmp1 = np.array(tmp0.strip().split(' '))
        tract_j = [float(numeric_string) for numeric_string in tmp1]
        length_j = int(len(tract_j)/3)
        length_all = np.concatenate((length_all, length_j), axis=0)
        tract_mat_j = np.transpose(np.reshape(tract_j, newshape=(length_j, 3)))
        tract_mat = np.concatenate((tract_mat, tract_mat_j), axis=1)
length_all_0 = length_all[2:, 0]
tract_mat_0 = tract_mat[3, 1:]
mat = {'length': length_all_0, 'tracts': tract_mat_0}
savemat(brainet_dir + '/' + 'trk_' + mode + str(n_clusters) + '.mat', mat)

n_nodes = int(np.sum(beta != 0))
node_coors = np.concatenate((mu[beta != 0, 0:3], mu[beta != 0, 3:6]), axis=0)
node_coors[:, 0] = node_coors[:, 0]
node_coors[:, 1] = node_coors[:, 1]
node_coors[:, 2] = node_coors[:, 2]
node_color = 4 * np.ones(shape=(2*n_nodes, 1))
node_size = np.ones(shape=(2*n_nodes, 1))
node_output = np.concatenate((node_coors, node_color, node_size), axis=1)
np.savetxt(brainet_dir + '/' + 'node_' + mode + str(n_clusters) + '.node', node_output[:, :])

edge_output = np.zeros(shape=(2*n_nodes, 2*n_nodes))
beta_nonzero = abs(beta[beta != 0])
for t in range(n_nodes):
    edge_output[t, n_nodes+t] = beta_nonzero[t]
    edge_output[n_nodes + t, t] = beta_nonzero[t]
np.savetxt(brainet_dir + '/' + 'edge_' + mode + str(n_clusters) + '.edge', edge_output[:, :])
