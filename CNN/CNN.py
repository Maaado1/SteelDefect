import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200,200)),
    transforms.ToTensor()
])

dataset=datasets.ImageFolder(
    root=r"D:\idea\project\NEU-CLS",
    transform=transform
)

dataloader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

X,y=next(iter(dataloader))

classes = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.linear=nn.Linear(64,6)
        self.pool=nn.MaxPool2d(2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

    def forward(self,X):
        X=self.relu(self.pool(self.conv1(X)))
        X=self.relu(self.pool(self.conv2(X)))
        X=self.avgpool(X)
        X=self.flatten(X)
        return self.linear(X)

net=CNN().to(device)

loss_=nn.CrossEntropyLoss()
lr=0.001
optimizer=torch.optim.Adam(net.parameters(),lr)
epochs=50
best_acc=0

for epoch in range(epochs):
    net.train()
    correct=0
    total=0
    for X,y in dataloader:
        X=X.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat=net(X)
        loss=loss_(net(X),y)
        loss.backward()
        optimizer.step()

        pred=y_hat.argmax(dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
    acc=correct/total
    print(f"循环{epoch+1}时准确率acc为{acc}")
    if acc>best_acc:
        best_acc=acc
        torch.save(net.state_dict(),"CNN_best_acc.pth")

net.load_state_dict(torch.load(r"D:\idea\project\CNN\CNN_best_acc.pth",map_location=device,weights_only=True))

testset=datasets.ImageFolder(
    root=r"D:\idea\project\Testset",
    transform=transform
)

testloader=DataLoader(
    testset,
    batch_size=32,
    shuffle=True
)

Tcorrect=0
Ttotal=0
net.eval()
with torch.no_grad():
    for TX,Ty in testloader:
        TX=TX.to(device)
        Ty=Ty.to(device)
        Ty_hat=net(TX)
        pred=Ty_hat.argmax(dim=1)
        Tcorrect+=(pred==Ty).sum().item()
        Ttotal+=Ty.size(0)
    Tacc=Tcorrect/Ttotal
print(f"测试准确率为{Tacc}")
#0.87777777777