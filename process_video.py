
import sys
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

'''
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
'''
ass = []

# torch.Size([40, 540, 960, 3])

for i in range(1,50):
	if i != 2:
		continue
	fv = "Video_Clips_trim/"+str(i)+".mp4"
	dt,_,_ = torchvision.io.read_video(fv)
	#dt = dt.permute(0, 3, 1, 2)
	#print(dt.shape)
	#for j in range(dt.shape[0]):
	#	imgj = dt[j,:,:,:]
	#	print(imgj.shape)
	#	save_image(imgj, 'data/'+str(j)+'.png')


