import os,torch
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from torch.utils.data import random_split
import torch.nn as nn


image_path='./VGG16--/flowers'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomCrop(100),                     # 随机裁剪 这句话删除了 就会报错 
    transforms.RandomHorizontalFlip(),              # 左右翻转
    transforms.ToTensor(),                           #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  # 均值方差归一化
])
##--------------------------------方法1----------------------------------------------
all_dataset=ImageFolder(image_path,transform=transform)   ###打包为dataset
len_dataset=len(all_dataset)        #获取数据集长度
train_len, val_len, test_len = round(0.8*len_dataset)-1,round(0.1*len_dataset),round(0.1*len_dataset)   #获取训练测试验证集数量
train_set, val_set, test_set = torch.utils.data.random_split(all_dataset,[train_len,val_len, test_len])

##--------------------------------方法2----------------------------------------------

train_dataset, test_dataset = random_split(
    dataset=all_dataset,
    lengths=[3021, 1296],
    generator=torch.Generator().manual_seed(0))


train_dataloader=DataLoader(train_dataset,batch_size=12,shuffle=True,num_workers=0)   ##打包为dataloader
test_dataloader=DataLoader(test_dataset,batch_size=12,shuffle=True,num_workers=0)




LEARNING_RATE = 0.001
EPOCH = 50
N_CLASSES = 5
total_train_step = 0
total_test_step = 0
#vgg16 = VGG16(n_classes=N_CLASSES)
vgg16=models.vgg16() # 直接API倒入vgg模型
vgg16.to(device)


# Loss, Optimizer & Scheduler
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
step=0
for epoch in range(EPOCH):
    print('————————第{}轮训练开始————————'.format(epoch+1))
    avg_loss = 0
    cnt = 0
    for data in train_dataloader:
        images,labels=data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss=cost(outputs,labels)
        # Forward + Backward + Optimize
        loss.backward()
        avg_loss += loss.data
        cnt += 1
        if step%100==0:
            print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        #loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)


# Test the model
vgg16.eval()
correct = 0
total = 0
total_test_loss=0
for images, labels in test_dataloader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = vgg16(images)
    loss=cost(outputs,labels)
    total_test_loss=total_test_loss+loss
    acc=(outputs.argmax(1)==labels).sum()
    correct = correct+acc
    predicted = torch.max(outputs.data, 1)
    #total += labels.size(0)
    #correct += (predicted == labels).sum()
    if step%100==0:
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100* correct/total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'cnn.pkl')
