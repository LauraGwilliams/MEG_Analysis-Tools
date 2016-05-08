# run MDS on MEG data
# Author: Laura Gwilliams (NYU)
# Email: leg5@nyu.edu
# Dependencies: scikitlearn, matplotlib
# Version: 1- 27/04/16

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from eelbrain import load, combine
from glob import glob
import numpy as np
from sklearn import manifold
from sklearn.utils import check_random_state
import time


tstart = 0
tstop = 0.5
step_size = 0.02
window_size = 0.04

# load epochs from pickle ( all files in given directory )
filedir = '/Volumes/LEG_2TB/Documents/Experiments/BP/barakeet_data/STG_TTG/dss_pickle/'
# make an empty bin to populate w/ data
dss = []
# loop through each pickle file in the directory
files = glob('%s*pickled' % filedir)
for f in files:
    ds = load.unpickle(f) # load up data
    ds['srcm'] = ds['srcm'].sub(time=(tstart,tstop)) # subset time of interest
    dss.append(ds) # add to bin
    del(ds) # delete to reduce redundancy (because already added to bin)
    print f # print file name

# combine all ptps into one dataset and pull out the data
ds = combine(dss)

# set var
var_of_interest = 'low_high_surp'

# aggregate over variable x subject
ds_agg = ds.aggregate('{}{}'.format('subject%', var_of_interest),drop_bad=True)

# load a label to subset
thisLbl = mne.read_labels_from_annot('fsaverage',parc='aparc',hemi='lh',regexp='transverse',subjects_dir='/Volumes/LEG_2TB/Documents/Experiments/BP/data/mri')[0]

# get out item names
try:
    item_names = np.array(map(str,ds_agg[var_of_interest].as_labels()))
except:
    item_names = np.array(map(int,ds_agg[var_of_interest].x))

# generate a colour for each unique variable
unique_items = set(item_names)
colours = np.random.rand(len(unique_items),3)

# loop through time windows
min_time_points = (tstart,tstop)
for lower_lim in np.arange(min_time_points[0], min_time_points[1]-step_size, step_size):
    dsX = ds_agg['srcm'].sub(source=thisLbl).sub(time=(lower_lim,lower_lim+window_size)).mean('source').x
    print dsX.shape, lower_lim

    # fit mds
    mds = manifold.MDS(2, max_iter=100, n_init=1, random_state=42, dissimilarity='euclidean')
    trans_data = mds.fit_transform(dsX).T

    # initiate figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # loop through and plot each variable
    i = 0
    for variable in unique_items:
        # bool to subset data from just var of interest
        boolean_list = np.array( ds_agg[var_of_interest] == str(variable))
        ds_var = ds_agg.sub( boolean_list ) # subset labels
        trans_data_sub = trans_data[:,boolean_list] # subset mds data

        plt.scatter(trans_data_sub[0], trans_data_sub[1], c=colours[i], label=str(variable), cmap=plt.cm.rainbow)
        i = i + 1

    ax.set_title('MDS: %s - %s ms' % (lower_lim*1000, (lower_lim+window_size)*1000))
    ax.legend(title='%s' % var_of_interest)
    plt.show(block=False)
    plt.draw()

    # wait a short amount of time and then delete plot
    time.sleep(1)
    plt.close()
