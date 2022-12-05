import os
from main2 import model 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print("In Attack file")
print("Downloading data ...")
test_csv = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  

test_loader = torch.utils.data.DataLoader(test_csv,
                                          batch_size=100)
print("done.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading model in attack file")
model.load_state_dict(torch.load(r"C:\Users\Mintu\OneDrive\Desktop\ASS 2 ML\ASS 2 ML\res_DC_FashionMNIST_ConvNet_1ipc.pt"))
print("done..")

''' Fast Gradient Signed Attack (fgsm) '''
def fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break
    
def plot_images(X,y,yp,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()


def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

print("Success Rate of FGSM : {}%".format(epoch_adversarial(model, test_loader, fgsm, 0.1)[0]*100))

''' Projected Gradient Descent  Attack '''

def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

# Illustrate original predictions


# Illustrate attacked images
delta = pgd(model, X, y, 0.1, 1e4, 1000)
yp = model(X + delta)
plot_images(X+delta, y, yp, 3, 6)

delta = torch.zeros_like(X, requires_grad=True)
loss = nn.CrossEntropyLoss()(model(X + delta), y)
loss.backward()

print("Success Rate of PGD : {}%".format(epoch_adversarial(model, test_loader, pgd, 0.1, 1e-2, 40)[0]*100))