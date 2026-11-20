import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

class Resblock(nn.Module):
    def __init__(self,input_channels,output_channels,use_1x1conv=False,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels, output_channels, 3,stride=stride,padding=1)
        self.conv2=nn.Conv2d(output_channels,output_channels,3,stride=1,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,output_channels,1,stride=stride)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(output_channels)
        self.bn2=nn.BatchNorm2d(output_channels)

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)
b1=nn.Sequential(
    nn.Conv2d(1,64,7,1,3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(3,2,1)
)
def RESBmaker(input_channels,output_channels,num,is_first=False):
    blk=[]
    for i in range(num):
        if i==0 and not is_first:
            blk.append(Resblock(input_channels, output_channels,use_1x1conv=True,stride=2))
        else:
            blk.append(Resblock(output_channels,output_channels))
    return blk
b2=nn.Sequential(*RESBmaker(64,64,2,is_first=True))
b3=nn.Sequential(*RESBmaker(64,128,2))
b4=nn.Sequential(*RESBmaker(128,256,2))
#b5=nn.Sequential(*RESBmaker(256,512,2))


net=nn.Sequential(
    b1,b2,b3,b4,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(256,6)
).to(device)

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
        loss=loss_(y_hat,y)
        loss.backward()
        optimizer.step()

        pred=y_hat.argmax(dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
    acc=correct/total
    print(f"循环{epoch+1}时准确率acc为{acc}")
    if acc>best_acc:
        best_acc=acc
        torch.save(net.state_dict(),"ResNet-b5_best_acc.pth")

net.load_state_dict(torch.load(r"D:\idea\project\ResNet\ResNet-b5_best_acc.pth",map_location=device,weights_only=True))

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
#Tacc=0.97777777777
#删除b5后训练时间明显缩短（8min25s），但准确率依然达到0.91944444