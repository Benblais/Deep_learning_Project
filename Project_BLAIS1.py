#!/usr/bin/env python
# coding: utf-8
#BLAIS Benjamin
#15/02/2018

# Import
import sys
import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


#Read folder
def read_folder(path):
    files = os.listdir(path)
    for name in files:
        if name.find(' ') != -1:
            os.rename(path+'/' + name, path+ '/' +name.replace(' ', '_'))



#Directory train and test
path_train="fruits-360/Training"
path_test="fruits-360/Test"
read_folder(path_train)
read_folder(path_test)

#Insert dataset
train_dataset = datasets.ImageFolder(path_train, transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

test_dataset = datasets.ImageFolder(path_test, transform = transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True)

#Init

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 40, 5)
        self.pooling = nn.MaxPool2d(2, 2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(40*22*22, 400)
        self.fc2 = nn.Linear(400,100)
        self.fc3 = nn.Linear(100,83)


    def forward(self,x):
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def __init__(self):
    #
    #     super(Net, self).__init__()
    #     # 1 input image channel, 6 output channels, 5x5 square convolution kernel
    #     self.conv1 = nn.Conv2d(3, 16, 5)
    #     self.conv2 = nn.Conv2d(16, 32, 5)
    #     self.conv3 = nn.Conv2d(32 ,64 ,5)
    #     self.conv4 = nn.Conv2d(64,128,5)
    #     self.pooling = nn.MaxPool2d(2, 2)
    #
    #     # an affine operation: y = Wx + b
    #     self.fc1 = nn.Linear(512, 1024) # (size of input, size of output)
    #     self.fc2 = nn.Linear(1024,256)
    #     self.fc3 = nn.Linear(256, 90)
    #
    #
    # def forward(self,x):
    #     x = self.pooling(F.relu(self.conv1(x)))
    #     x = self.pooling(F.relu(self.conv2(x)))
    #     x = self.pooling(F.relu(self.conv3(x)))
    #     x = self.pooling(F.relu(self.conv4(x)))
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x



    def num_flat_features(self, x):
        size=x.size()[1:]
        # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features
#check CPU on machine
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
if str(device)=='cuda:0':
    print ("You use cuda")
    print ('\n')
else:
    print ('You use CPU')
    print('\n')
#For use cuda
net = Net().to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


epochs = 4
for epoch in range(epochs):
    running_loss=0.0
    for i, data in enumerate(train_loader,0):
        inputs,labels = data
        inputs,labels = inputs.to(device), labels.to(device) # for cuda
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % 2000 == 1999: #printevery 2000 mini-batches
            print ('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print ('\n')
print ('Training phase is finished!')
print ('\n')

#Test phase
testiter = iter(test_loader)
images, labels =  testiter.next()
print (labels)

images = images.to(device, torch.float)
outputs = net(images)
_,predicted = torch.max(outputs,1)
print (predicted)
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device, torch.float)
        labels = labels.to(device, torch.long)
        outputs=net(images)
        _,predicted= torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print ('\n')
print ('Test phase is finished!')
print ('\n')
print ('Number of correct images :', correct)
print ('Number of total images : ', total)
print('\n')
print ('Accuracy of the network on the 14 369 test images is: %d %%' %(100* correct / total))
