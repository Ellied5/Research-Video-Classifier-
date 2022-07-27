import sys
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.io import write_jpeg
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
import random
import os 


data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
trans=transforms.ToPILImage()
ass = []

torch.Size([40, 540, 960, 3])
print("Processing Post-Injury Videos..")
for i in range(1,171):
	file="T ("+str(i)+").mp4"
	fv ="Individual_clips/Post-Injury/"+file 
	dt,_,_ = torchvision.io.read_video(fv)
	dt = dt.permute(0, 3, 1, 2)
	q1=int(dt.shape[0]*.25)
	q3=int(dt.shape[0]*.75)
	l = random.sample(range(q1,q3), 150)
	for k in range(int(len(l)/10)): 
		for j in range(10):
			imgj = dt[l[j+k*10],:,:,:]
			imgj=trans(imgj)
			#imgj=data_transforms(imgj)
			#imgj=trans(imgj)
			imgj=transforms.CenterCrop(250)(imgj)
			imgj=transforms.Grayscale()(imgj)
			imgj=transforms.ToTensor()(imgj)  
			newpath = 'data/Post-Injury/'+str(i)+str(k)+'/' 
			if not os.path.exists(newpath):
				os.makedirs(newpath)
			save_image(imgj, newpath+str(j)+'.tiff')
print("Completed.")

print("Processing Pre-Injury Videos..")   
for i in range(1,663):  
	file="R("+str(i)+").mp4"
	fv ="Individual_clips/Pre-Injury/"+file 
	dt,_,_ = torchvision.io.read_video(fv)
	dt = dt.permute(0, 3, 1, 2)
	q1=int(dt.shape[0]*.25)
	q3=int(dt.shape[0]*.75)
	l = random.sample(range(q1,q3), 50)
	for k in range(int(len(l)/10)): 
		for j in range(10):
			imgj = dt[l[j+k*10],:,:,:]
			imgj=trans(imgj)
			#imgj=data_transforms(imgj)
			#imgj=trans(imgj)
			imgj=transforms.CenterCrop(250)(imgj)
			imgj=transforms.Grayscale()(imgj)
			imgj=transforms.ToTensor()(imgj)    
			newpath = 'data/Pre-Injury/'+str(i)+str(k)+'/'
			if not os.path.exists(newpath):
				os.makedirs(newpath)
			save_image(imgj, newpath+str(j)+'.tiff')
print("Completed.")
