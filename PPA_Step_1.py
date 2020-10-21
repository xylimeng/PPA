"""
PPA Step-1: extract the ending points from fiber tracking results

Author: RJ Liu (rl58@rice.edu)
Last update: 2020-04-04
"""

import numpy as np
import pandas as pd
import os
import fnmatch
from scipy.io import loadmat, savemat

"""
installed all the libraries above
"""

home_dir = './'
ft_file_dir = home_dir + 'data'
output_dir = home_dir + 'ending_point'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# screen all fiber tracking files
sub_id_list = np.loadtxt(home_dir + 'subjects_0.txt')
n = len(sub_id_list)

print('load the fiber tracking file \n')
n_tracts = np.zeros(shape=(n, 1))
for k in range(n):
    print(sub_id_list[k])
    file_list = fnmatch.filter(os.listdir(ft_file_dir), str(int(sub_id_list[k])) + '*.txt')
    file_name = file_list[0]
    file_path = ft_file_dir + '/' + file_name
    sub_id = file_name[:7]

    dat = pd.read_csv(file_path, header=None)
    n_tract = len(dat)
    n_tracts[k, 0] = n_tract
    end_points_new = np.zeros(shape=(n_tract, 6))
    for j in range(n_tract):
        tmp0 = dat.loc[j][0]
        tmp1 = np.array(tmp0.strip().split(' '))
        tract_j = [float(numeric_string) for numeric_string in tmp1]
        end_point_1 = tract_j[:3]
        end_point_2 = tract_j[-3:]
        if end_point_1[2] > end_point_2[2]:
            tmp = end_point_2
            end_point_2 = end_point_1
            end_point_1 = tmp
        end_points_new[j, 0] = 78 - end_point_1[0]
        end_points_new[j, 1] = 76 - end_point_1[1]
        end_points_new[j, 2] = end_point_1[2] - 50
        end_points_new[j, 3] = 78 - end_point_2[0]
        end_points_new[j, 4] = 78 - end_point_2[1]
        end_points_new[j, 5] = end_point_2[2] - 50
    mat = {'end_points': end_points_new}
    savemat(output_dir + '/' + sub_id + '_end_points.mat', mat)
np.savetxt(output_dir + '/' + 'n_tracts.txt', n_tracts[:, 0])
