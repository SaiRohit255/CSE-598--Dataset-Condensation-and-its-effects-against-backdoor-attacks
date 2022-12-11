# CSE-598 Dataset Condensation and its effects against backdoor attacks

**Introduction**<br><br/>
Devices have become powerful enough to run large and 
complex problems in the modern world, and we are using 
increasingly large datasets day by day for developing 
newer machine learning models. In a world in which 
machine learning has become an everyday part of our 
lives, these advancements have been made possible by the 
fact that machine learning has become an integral part of 
our society. The process of data condensing helps a great 
deal when it comes to reducing the amount of processing 
power needed to utilize these massive datasets since it 
may lower the size of a dataset while still producing 
findings that are comparable to those of the original 
dataset (which is not condensed.)
<br><br/>
The goal of this study is to determine whether condensing 
datasets help make machine learning models that are 
vulnerable to backdoor attacks more resilient. 
Checking the performance of a deep neural network that 
is trained using a condensed dataset by data condensation 
techniques is the primary objective of this project. 
Specifically, the goal is to determine whether or not this 
model performs more reliably than a standard model that 
is trained using a normal dataset (a dataset that is not 
condensed).

**Dataset condensation**<br><br/>
Condensing a large training set into a smaller synthetic set is the goal of dataset condensation. This is done so that the model trained on the smaller synthetic set may achieve testing performance that is equivalent to that of the model trained on the larger training set.<br><br/>

**Setup**<br><br/>
1. We used PyCharm for our project.<br><br/>
2. CONDA as our Interperter or Environment.<br><br/>
3. Install packages in the requirements.

**Procedure**<br><br/>
First, we analyze the performance of the classification 
system using the condensed images on the standard 
benchmark dataset, which is the FashionMNIST dataset 
for classification tasks. We put our technique through its 
pace by using a conventional deep network design known 
as ConvNet. ConvNet is a modular architecture that is 
frequently utilized in few-shot learning. It consists of D 
duplicate blocks, and each block possesses the following: 
a convolutional layer with W (3x3) filters, a 
normalization layer N, an activation layer A, and a 
pooling layer P, which is represented by the notation [W, 
N, A, P]xD. If nothing else is supplied, the default 
configuration of the ConvNet consists of three blocks, 
each of which has 128 filters, followed by modules for 
InstanceNorm, ReLU, and AvgPooling. A linear classifier 
comes after the very last block in the sequence. Kaiming 
is the initialization that we utilize for the network weights. 
Initialization of the synthetic images may be 
accomplished either via the use of Gaussian noise or by 
the random selection of genuine training images.<br><br/>
We investigated the test performance of a ConvNet that 
was trained on them for the FashionMNIST dataset for a 
variety of numbers of condensed images for each class, 
both in absolute and relative terms, with the results being 
normalized by their upper bound. Increasing the 
number of condensed images results in improved 
accuracy across all benchmarks and reduces the gap 
between the current performance and the upper-bound 
performance in FashionMNIST.<br><br/>
After condensing the FashionMNIST dataset we attacked 
the Lenet model which was trained on the condensed 
dataset with a fast gradient sign method attack the attack 
accuracy of the backdoor attack and loss vs iteration data 
are given below for both models which are trained on the 
original and condensed dataset.
