"""
PPA Step-2: mini_batch K-means on ending points

Author: RJ Liu (rl58@rice.edu)
Last update: 2020-04-04
"""

import numpy as np
import os
import sys
import fnmatch
from scipy.io import loadmat
from sklearn.cluster import MiniBatchKMeans

"""
installed all the libraries above
"""

home_dir = './'
input_dir = home_dir + 'ending_point'
output_dir = home_dir + 'clustering'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
n_clusters = int(sys.argv[1])

# screen all fiber tracking files
sub_id_list = np.loadtxt(home_dir + 'subjects_0.txt')
n = len(sub_id_list)

n_tracts = np.loadtxt(input_dir + '/' + 'n_tracts.txt')
n_tract_all = int(np.sum(n_tracts))
# print(n_tract_all)
end_points_all = np.zeros(shape=(n_tract_all, 6))

print('load the ending point file \n')
idx_1 = 0
idx_2 = n_tracts[0]
for k in range(n-1):
    file_list = fnmatch.filter(os.listdir(input_dir), str(int(sub_id_list[k])) + '*.mat')
    file_name = file_list[0]
    file_path = input_dir + '/' + file_name
    # sub_id = file_name[:-4]

    mat = loadmat(file_path)
    end_points = mat['end_points']
    # print(end_points.shape)
    end_points_all[int(idx_1):int(idx_2), :] = end_points
    idx_1 = idx_1 + n_tracts[k]
    idx_2 = idx_2 + n_tracts[k + 1]
    del end_points

kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
# Fitting the input data
kmeans = kmeans.fit(end_points_all)
labels = kmeans.predict(end_points_all)
cluster_means = kmeans.cluster_centers_

w = np.zeros(shape=(n, n_clusters))
idx_1 = 0
idx_2 = n_tracts[0]
labels_i = labels[int(idx_1):int(idx_2)]
for j in range(n_clusters):
    w[0, j] = np.mean(1*(labels_i == j))
for i in range(n-1):
    idx_1 = idx_1 + n_tracts[i]
    idx_2 = idx_2 + n_tracts[i + 1]
    labels_i = labels[int(idx_1):int(idx_2)]
    for j in range(n_clusters):
        w[i+1, j] = np.mean(1*(labels_i == j))

np.savetxt(output_dir + '/' + 'cluster_w_' + str(n_clusters) + '.txt', w)
np.savetxt(output_dir + '/' + 'cluster_label_' + str(n_clusters) + '.txt', labels)
np.savetxt(output_dir + '/' + 'cluster_means_' + str(n_clusters) + '.txt', cluster_means)
