import os
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
import sys


for i in range(1,50):
	if i != 1:
		continue
	dirin = "images/"+str(i)+"/"
	dirout = "images_grey/"+str(i)+"/"
	if not os.path.exists(dirout):
		os.makedirs(dirout)
	
	print(len(listdir(dirin)))
	for c in sorted(listdir(dirin)):
		img_path = join(dirin, c)
		img_path2 = join(dirout, c)
		#if str(img_path)[-4:] == "tiff":
			#print(c)
			#img = Image.open(sys.argv[1]).convert('L').resize((256, 256))
			#img = Image.open(img_path).convert('L')
			#img.save(img_path2)


#img = np.array(img, dtype=np.float32) / 256.0
#img = np.array(img, dtype=np.float32)
#print(img[0,:])
#print(img.shape)



