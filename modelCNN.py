import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def progress_bar(progress, total):
    percent = 100*(progress/float(total))
    bar = '■' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


size_x = 32
size_y = 32
threshold = .2
test_size = .2
batch_size = 16
learning_rate = .001
weight_decay = 1e-5

# Number of noise samples added to the dataset
amount_of_noise = 100

transform = transforms.Compose([transforms.Resize((size_x, size_y)),
                                transforms.ToTensor()])
folder = datasets.ImageFolder('Data/Letters', transform=transform)


number_of_symbols = len(folder.class_to_idx)

data = torch.ones((len(folder) + amount_of_noise, 1, size_x, size_y))
labels = torch.zeros(len(folder) + amount_of_noise).long()


##########################
# Processing training data
##########################


print("Processing Data")


for img in range(len(folder)):
    progress_bar(img + 1, len(folder))
    for i in range(size_x):
        for j in range(size_y):
            if folder[img][0][0, i, j] > threshold:
                data[img][0, i, j] = 0
    labels[img] = folder[img][1]

for i in range(amount_of_noise):
    noise = torch.randint(0, 2, (size_x, size_y))
    data[len(folder) + i] = noise
    labels[len(folder) + i] = number_of_symbols
number_of_symbols += 1


print("\nFinished Data Processing\n")


test_data, train_data, test_labels, train_labels = train_test_split(data, labels, test_size=test_size)
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)


class LettersCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bnorm3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(2 * 2 * 256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, number_of_symbols)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.leaky_relu(self.bnorm1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu(self.bnorm2(x))
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.leaky_relu(self.bnorm3(x))

        x = x.view(-1, 2 * 2 * 256)

        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=.5, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=.5, training=self.training)
        x = self.fc3(x)

        return x


numepochs = 200
net = LettersCNN()


def train():
    print("\nTraining the model")
    net.to(device)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainAcc = []
    testAcc = []
    for epoch_i in range(numepochs):
        net.train()
        progress_bar(epoch_i + 1, numepochs)
        batchAcc = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            yHat = net(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
        trainAcc.append(np.mean(batchAcc))

        net.eval()
        X, y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)
        testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
    print("\nFinished Training\n")
    return trainAcc, testAcc


trainAcc, testAcc = train()

plt.plot(trainAcc)
plt.plot(testAcc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

model = torch.jit.script(net)
model.save('modelCNN3.pt')

print("Model Saved")
