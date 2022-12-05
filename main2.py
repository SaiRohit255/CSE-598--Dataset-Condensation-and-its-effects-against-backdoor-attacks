import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("loading data ..")

train_csv = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_csv = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  

train_loader = torch.utils.data.DataLoader(train_csv, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_csv,
                                          batch_size=100)
print("done")

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

a = next(iter(train_loader))
a[0].size()
torch.Size([100, 1, 28, 28])
# len(train_csv)

image, label = next(iter(train_csv))


demo_loader = torch.utils.data.DataLoader(train_csv, batch_size=10)

batch = next(iter(demo_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


model = FashionCNN()

model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 6
count = 0
# Lists for visualization of loss and accuracy 
loss_list = []
loss_list1=[]
iteration_list = []
accuracy_list = []
accuracy_list1 = []
count1= []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
    
        count += 1
    
    # Testing the model
        if not (count % 50): 
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
            
                test = Variable(images.view(100, 1, 28, 28))
            
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
         
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
            loss_list1.append(loss.item())
            count1.append(count)
            accuracy_list1.append(accuracy.item())

xpoints = np.array(count1)
ypoints = np.array(loss_list1)
plt.plot(xpoints,ypoints)
plt.title(' x - Iteration   y - loss values ')
plt.show()
#xpoints = np.array(count1)
ypoints = np.array(accuracy_list1)
plt.plot(xpoints,ypoints)
plt.title(' x - Iteration   y - Accuracy in percent ')
plt.show()


PATH = "C:/Users/apoogali/Desktop/Project"
torch.save(model.state_dict(), os.path.join(PATH,'FashionMNIST.pth'))