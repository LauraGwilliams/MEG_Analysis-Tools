# run PCA on MEG data
# Author: Laura Gwilliams (NYU)
# Email: leg5@nyu.edu
# Dependencies: scikitlearn, matplotlib, MNE-Python
# Version: 2- 08/05/16

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import mne
import numpy as np

# set up file directories to example MNE-Python data
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# determine size of epochs
tmin, tmax = -0.0, 0.4
event_id = dict(aud_l=1, vis_l=3)

# load data and make epochs
raw = mne.io.Raw(raw_fname, preload=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, preload=True, baseline=None)

# pull out the data, average over time, and get data labels
X = np.mean(epochs._data,2)
y = epochs.events[:,2]

# fit and apply the PCA
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

#-# plot 3D #-#

# set up dictionaries for plotting colour and legend labels
colour_dict = {'1':'r','3':'b'}
leg_dict = {'1':'Auditory','3':'Visual'}

# init figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# loop through each epoch and plot on 3D scatter
for label_idx in xrange(0,len(y)):
    ax.scatter(X_r[label_idx,0],X_r[label_idx,1],X_r[label_idx,2], marker='^',
               c=colour_dict.get(str(y[label_idx])), s=40,
               label=leg_dict.get(str(y[label_idx])) if label_idx < 2 else "_nolegend_")

# add legend and show
plt.legend(loc='upper left')
plt.title('PCA: Auditory vs. Visual Stimuli')
ax.set_xlabel('First PC'), ax.set_ylabel('Second PC'), ax.set_zlabel('Third PC')
plt.show()
