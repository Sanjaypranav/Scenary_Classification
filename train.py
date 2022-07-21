import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os
import torchvision
from torch.utils.data import  DataLoader
from torch import optim
from customDataset import ScenaryDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparameters
batch_size = 32
num_epochs = 5
learning_rate = 0.01
num_classes = 6
in_channels = 3
classes = {
   0 : "Buildings",
1: "Forests",
2: "Mountains" ,
3 : "Glacier" , 
4 : "Street",
5 : "Sea"
}
#Load the dataset
# Scenary_Classificationo/train.py
# Scenary_Classificationo/train-scene classification/train.csv

train_dataset = ScenaryDataset(csv_file='/media/senju/1.0 TB Hard Disk/UniversitiOfMalaya_Computer_Vision_Research/Scenary_Classification(Pytorch)/train-scene classification/train.csv', root_dir='/media/senju/1.0 TB Hard Disk/UniversitiOfMalaya_Computer_Vision_Research/Scenary_Classification(Pytorch)/train-scene classification/train/', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomHorizontalFlip(p=0.5),transforms.Resize((150,150))]))

train_set , test_set = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Model Definition
class ScenaryClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #Define the layers of the model
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        #Max pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #Flatten the output of the convolutional layers
        #flatten the output of conv4
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(256*7*7, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = ScenaryClassifier().to(device)

#Define the loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    #Test the model

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


#Save the model
torch.save(model.state_dict(), 'model.ckpt')
print("Model saved")


#Load the model
model.load_state_dict(torch.load('model.ckpt'))
print("Model loaded")

#PREDICTION FOR A NEW IMAGE from the test set
#Load the image
img = torch.randn(1, 3, 150, 150)
img = img.to(device)
#Forward pass
outputs = model(img)
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', classes[predicted.item()])
