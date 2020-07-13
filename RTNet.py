#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# In[2]:


import torch
import torch.nn as nn
import glob
from torch.utils.data.dataset import Dataset 
from PIL import Image
from torchvision import transforms,models
import matplotlib.pyplot as plt
import numpy as np
import random
#from torchsummary import summary
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

image_height=int(1200/4)
image_width=int(1600/4)


# In[3]:


class datasource(object):
    def __init__(self, images1, images2, poses, idx, max_size, categ):
        self.images1 = images1
        self.images2 = images2
        self.poses = poses
        self.max_size = max_size
        self.idx = idx
        self.pos = 0
        self.categ = categ
    def __len__(self):  # return count of sample we have
        return self.max_size
    
def get_data(mode = 'train'):
    poses = []
    images1 = []
    images2 = []
    categ = []

    with open('/home/ubuntu/DTU_Dataset/train_data_mvs.txt') as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            imgFiledId1, imgFiledId2, categoryId, x, y, z, q1, q2, q3, q4 = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            q1 = float(q1)
            q2 = float(q2)
            q3 = float(q3)
            q4 = float(q4)
            poses.append((x,y,z,q1,q2,q3,q4))
            categ.append(int(categoryId))
            imgFiledId1 = '0'+imgFiledId1 if len(imgFiledId1)==1 else imgFiledId1
            imgFiledId2 = '0'+imgFiledId2 if len(imgFiledId2)==1 else imgFiledId2
            images1.append('/home/ubuntu/DTU_Dataset/Cleaned' + '/scan'+categoryId+'/'                           +'clean_0' + imgFiledId1 + '_0_r5000.png')
            images2.append('/home/ubuntu/DTU_Dataset/Cleaned' + '/scan'+categoryId+'/'                           +'clean_0' + imgFiledId2 + '_0_r5000.png')
    max_size = len(poses)
    indices = list(range(max_size))
    #random.shuffle(indices)
    return datasource(images1, images2, poses, indices, max_size, categ)

def get_data_batch(source, batch_size):
    image1_batch = []
    image2_batch = []
    image1_path = []
    image2_path = []
    pose_x_batch = []
    pose_q_batch = []
    for i in range(batch_size):
        pos = i + source.pos
        pose_x = source.poses[source.idx[pos]][0:3]
        pose_q = source.poses[source.idx[pos]][3:7]
        image1_path.append(source.images1[source.idx[pos]])
        image2_path.append(source.images2[source.idx[pos]])
        image1=cv2.imread(source.images1[source.idx[pos]],cv2.IMREAD_COLOR)
        image2=cv2.imread(source.images2[source.idx[pos]],cv2.IMREAD_COLOR)
        image1_batch.append(cv2.resize(image1,dsize=(image_width,image_height),interpolation=cv2.INTER_AREA))
        image2_batch.append(cv2.resize(image2,dsize=(image_width,image_height),interpolation=cv2.INTER_AREA))
        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)
        
    if batch_size==1:
        source.pos+=1
    else:
        source.pos += i
    
    if source.pos + i > source.max_size:
        source.pos = 0
        
    image1_batch=torch.FloatTensor(np.array(image1_batch)).permute(0, 3, 1, 2)
    image2_batch=torch.FloatTensor(np.array(image2_batch)).permute(0, 3, 1, 2)
    pose_x_batch=torch.FloatTensor(np.array(np.array(pose_x_batch)))
    pose_q_batch=torch.FloatTensor(np.array(np.array(pose_q_batch)))
                                   
    return image1_batch, image2_batch, pose_x_batch, pose_q_batch
                                   


# In[4]:


dataset=get_data()

images1_train,images1_test= train_test_split(dataset.images1, test_size=0.2, random_state=321)
images2_train,images2_test= train_test_split(dataset.images2, test_size=0.2, random_state=321)
poses_train,poses_test= train_test_split(dataset.poses, test_size=0.2, random_state=321)
indices_train,indices_test= train_test_split(dataset.idx, test_size=0.2, random_state=321)
categ_train,categ_test= train_test_split(dataset.categ, test_size=0.2, random_state=321)

train_indices = list(range(len(images1_train)))
random.shuffle(train_indices)

test_indices = list(range(len(images1_test)))
random.shuffle(test_indices)

train_data=datasource(images1_train,images2_train,poses_train,train_indices,len(poses_train),categ_train)
test_data=datasource(images1_test,images2_test,poses_test,test_indices,len(poses_test),categ_test)

#image1,image2,pose_x,pose_y = get_data_batch(train_data,30)

#print(len(image1))
#print(pose_x[0].shape)
#print(pose_y[0].shape)

#print(image1[0][0],image1[0][0])


batch_size=16


# In[5]:


class RTNet(nn.Module):
    def __init__(self):
        super(RTNet, self).__init__()
        
        googlenet1 = models.googlenet(pretrained=True)
        
        self.googlenetL =nn.Sequential(*list(googlenet1.children())[:-3]) 
        self.googlenetR =nn.Sequential(*list(googlenet1.children())[:-3]) 
        
        for param in self.googlenetL.parameters():
            param.require_grad = False
        for param in self.googlenetR.parameters():
            param.require_grad = False
            
        self.convL=torch.nn.Conv2d(1024,30,1)
        self.convR=torch.nn.Conv2d(1024,30,1)
        
        self.linear_1 = torch.nn.Linear(7020, 1024)
        self.linear_2 = torch.nn.Linear(1024,7)
        self.relu = torch.nn.ReLU()
        
        
    def forward(self, imageL,imageR):
        
        x=self.googlenetL(imageL)
        y=self.googlenetR(imageR)
        
        x=self.convL(x)
        y=self.convR(y)
        
        x = x.view(-1, 3510)
        y = y.view(-1, 3510)
        
        result=torch.cat((x,y),1)
        
        result=self.relu(self.linear_1(result))
        result=self.linear_2(result)
        
        return result


# In[6]:


model=RTNet().cuda()
#print(model)

use_cuda=1

if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.MSELoss()

#summary(model, [(3, 1600, 1200), (3, 1600, 1200)])


# In[ ]:


numEpochs=5

train_loss = list()
val_loss = list()
accu = []


iterate=0
averageLoss=0
batchIter=int(len(train_data)/batch_size)
print(batchIter)

for epoch in range(0,numEpochs):
    total_val_loss=0
    model.train()
    
    for batch_idx in range(batchIter):
        
        image1,image2,pose_x,pose_y = get_data_batch(train_data,batch_size)
        
        target=torch.cat((pose_x,pose_y),1)
        
        image1 = image1.cuda(async=True)
        image2 = image2.cuda(async=True)
        target = target.cuda(async=True)
        
        optimizer.zero_grad()
        output = model(image1,image2)
        #print('target : ', target[0].detach().cpu().numpy(),'\noutput : ',output[0].detach().cpu().numpy())
        
        loss = criterion(output, target)
        
        averageLoss+=loss.data
        
        train_loss.append(loss.data)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx%10==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx , batchIter,
                100. * batch_idx / batchIter, averageLoss/10))
            averageLoss=0

        iterate=iterate+1
            #if loss.item()<0.1:
            #    break
    


# In[ ]:


fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, iterate+1), train_loss, label="Train loss")
#plt.plot(np.arange(1, numEpochs+1), val_loss, label="Validation loss")
plt.xlabel('iterate')
plt.ylabel('Loss')
plt.title("Loss Plots")
plt.legend(loc='upper right')
#plt.show()
#plt.savefig('loss.png')


# In[ ]:


#torch.save({
#        'epoch': epoch,
#        'model_state_dict': model.state_dict(),
#        'optimizer_state_dict': optimizer.state_dict(),
#        }, './RTN_googleNet.pth')


# In[ ]:


model.eval()
    
for epoch in range(0,1):
    
    test_loss = 0
    correct = 0
    averageLoss=0
    
    for batch_idx in range(1): 
        
        image1,image2,pose_x,pose_y = get_data_batch(test_data,1)
        #print(pose_x[0],pose_x.shape)
        
        print(image1_path)
        print(image2_path)
        
        plt.figure()
        plt.imshow(image1[0,0,:,:],cmap='gray')
        plt.figure()
        plt.imshow(image2[0,0,:,:],cmap='gray')
        
        target=torch.cat((pose_x,pose_y),1)
        
        image1 = image1.cuda(async=True)
        image2 = image2.cuda(async=True)
        target = target.cuda(async=True)
        
        optimizer.zero_grad()
        output = model(image1,image2)
        
        print('target : ', target[0].detach().cpu().numpy(),'\noutput : ',output[0].detach().cpu().numpy())
        
        loss = criterion(output, target)
        
        averageLoss+=loss.data
        
        
    print('\nTest set: Average loss: {:.4f}'.format(
        averageLoss/1))


# In[ ]:




