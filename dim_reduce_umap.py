
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold, datasets, cluster, mixture
import umap
# 245 = 49*5
# 245*10*32*32*32
# 1-17 before
# 17*5 = 85 before
# 6*5 = 30 after one week
# 10*5 = 50 after two weeks
# 8*5 = 40 after three weeks
# 8*5 = 40 after four weeks
dat = np.load(sys.argv[1])

#dor = int(sys.argv[3])
dor = 9

nall = dat.shape[0]
dat2 = dat[:,dor,:,:,:]
dat2 = dat2.reshape(nall, -1)

##### labels
labels = np.concatenate((np.zeros(85), np.ones(245-85)))
labels2 = np.concatenate((np.zeros(85), np.ones(30), 2*np.ones(50), 3*np.ones(40), 4*np.ones(40)))
##### umap
#for nneigh in (2, 5, 10, 20, 50, 100, 200):
for nneigh in range(15,100,5):
	reducer = umap.UMAP(n_components=3,n_neighbors=nneigh)
	dat_umap = reducer.fit_transform(dat2)
	dat_umap2 = np.concatenate((dat_umap, labels.reshape(nall,-1), labels2.reshape(nall,-1)), axis=1)
	np.savetxt(sys.argv[2]+"_umap"+str(nneigh), dat_umap2, fmt='%1.5f')



