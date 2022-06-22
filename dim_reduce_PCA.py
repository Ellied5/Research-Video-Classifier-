
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold, datasets, cluster, mixture

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
##### PCA
'''
pca = PCA(n_components = 3)
dat_PCA = pca.fit_transform(dat2)
dat_PCA2 = np.concatenate((dat_PCA, labels.reshape(nall,-1), labels2.reshape(nall,-1)), axis=1)
np.savetxt(sys.argv[2]+"_PCA", dat_PCA2, fmt='%1.5f')
'''

##### tSNE
#perplexities = [45, 50, 55]
#for i, perplexity in enumerate(perplexities):
for perplexity in range(5,56,5):
	tsne = manifold.TSNE(n_components=3, init='random', n_iter=5000,
                         random_state=0, perplexity=perplexity)
	X = tsne.fit_transform(dat2)
	fout = sys.argv[2]+"_tsne"+str(perplexity)
	X2 = np.concatenate((X, labels.reshape(nall,-1), labels2.reshape(nall,-1)), axis=1)
	np.savetxt(fout, X2, fmt='%1.5f')


