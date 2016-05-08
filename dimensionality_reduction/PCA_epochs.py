# run ICA on MEG data
# Author: Laura Gwilliams (NYU)
# Email: leg5@nyu.edu
# Dependencies: scikitlearn, matplotlib
# Version: 1- 26/04/16

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from eelbrain import load, combine
from glob import glob
import numpy as np

# load epochs from pickle ( all files in given directory )
filedir = '/Volumes/LEG_2TB/Documents/Experiments/BP/barakeet_data/STG_TTG/dss_pickle/'
dss = []
files = glob('%s*pickled' % filedir)
for f in files:
    ds = load.unpickle(f)
    # get the data down to 2 dims for the PCA
    ds['srcm'] = ds['srcm'].sub(time=(0.0,0.2)).mean('time')
    dss.append(ds)
    del(ds)
    print f

# combine all ptps into one dataset and pull out the data
ds = combine(dss)
thisBool = ds['POD_distance_ms'] > 450
ds = ds.sub(np.array(thisBool))
dsX = ds['srcm'].x

# fit and apply the PCA
pca = PCA(n_components=3)
X_r = pca.fit(dsX).transform(dsX)

# plot 2D
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_r[:,0],X_r[:,1])
plt.show()

# plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_r[:,0],X_r[:,1],X_r[:,2])
plt.show()
