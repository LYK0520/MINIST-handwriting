import random
import torch
import torchvision
import scipy.io
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


import matplotlib.pyplot as plt

print(torch.cuda.is_available())

# 读取 .mat 文件
data = scipy.io.loadmat('handwriting.mat')

print("Data Keys:", data.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])

train_imgs=data['train_imgs']
train_labels=data['train_labels']   
test_imgs=data['test_imgs']
test_labels=data['test_labels']



train_imgs = train_imgs.transpose((2, 0, 1))  # 改变维度顺序到 (num_samples, 28, 28)
test_imgs = test_imgs.transpose((2, 0, 1))  # 改变维度顺序到 (num_samples, 28, 28)

train_labels=train_labels.flatten()
test_labels=test_labels.flatten()

print("Train Images Shape:", train_imgs.shape)  # (60000, 28, 28)
print("Train Labels Shape:", train_labels.shape)  # (60000,)
print("Test Images Shape:", test_imgs.shape)    # (10000, 28, 28)
print("Test Labels Shape:", test_labels.shape)    # (10000,)

{
    # # 提取训练数据和标签
# images = data['X']  # 形状为 (28, 28, num_samples)
# labels = data['y'].flatten()  # 转换成一维数组

# # 转置和 reshape 为 PyTorch 可以使用的格式
# images = images.transpose((2, 0, 1))  # 改变维度顺序到 (num_samples, 28, 28)
# images = torch.tensor(images, dtype=torch.float32)  # 转换为张量
# labels = torch.tensor(labels, dtype=torch.long)  # 转换为张量

# # 按需标准化（例如，缩放到 [0, 1]）
# images /= 255.0

# # 定义训练集和测试集的划分
# num_train = 60000  # MNIST 的训练集样本数量
# train_images = images[:num_train]
# train_labels = labels[:num_train]
# test_images = images[num_train:]
# test_labels = labels[num_train:]

# # 验证数据形状
# print("Train Images Shape:", train_images.shape)  # (60000, 28, 28)
# print("Train Labels Shape:", train_labels.shape)  # (60000,)
# print("Test Images Shape:", test_images.shape)    # (10000, 28, 28)
# print("Test Labels Shape:", test_labels.shape)    # (10000,)
}



n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].astype(np.float32)
        label = self.labels[index]


        
        # 应用变换
        if self.transform:
            image = self.transform(image)

        

        return image, label

# 创建 DataLoader
transform = torchvision.transforms.Compose([  torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])  

train_dataset = MNISTDataset(train_imgs, train_labels,transform=transform)
test_dataset = MNISTDataset(test_imgs, test_labels, transform=transform)



train_loader =torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test,shuffle=False)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 打印输入形状
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(f"After conv1: {x.shape}")
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(f"After conv2: {x.shape}")
        x = x.view(x.size(0), -1)  # 自动推导出第一个维度为批次大小
        #print(f"After view: {x.shape}")
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #print(f"Output shape: {x.shape}")
        return F.log_softmax(x, dim=1)
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    # print(f"Target shape: {target.shape}")
    # print(f"Target values: {target}")
    optimizer.zero_grad()
    output = network(data)
    # print(f"Output shape: {output.shape}")
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
# 加载模型和优化器
def load_model(model_path, optimizer_path):
    model = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    # 加载模型
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # 切换到评估模式
    
    # 加载优化器状态
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    
    return model, optimizer

def pic_show():



    # 加载模型和优化器
    network, optimizer = load_model('results/model.pth', 'results/optimizer.pth')
    
    # 假设你已经有了加载的模型和 test_loader
    #network.eval()  # 将模型设为评估模式

    # 随机从测试集中选择六张图片
    indices = random.sample(range(len(test_dataset)), 100)
    images, labels = zip(*[test_dataset[i] for i in indices])

    # 将数据转换为张量，并传入模型
    images_tensor = torch.stack(images)
    with torch.no_grad():  # 不需要计算梯度
        outputs = network(images_tensor)

    # 获取预测结果
    _, predicted = torch.max(outputs.data, 1)

    # 展示图片和预测标签
    plt.figure(figsize=(12, 10))  # 设置合适的画布大小
    for i in range(100):
        plt.subplot(10, 10, i + 1)  # 创建 10x10 的子图
        if predicted[i].item() != labels[i]:  # 判断预测和真实值是否不同
            plt.imshow(images[i].squeeze(), cmap='gray')  # 显示图片
            plt.title(f'Pred: {predicted[i].item()}, True: {labels[i]}', fontsize=8, color='red')  # 标红标题
        else:
            plt.imshow(images[i].squeeze(), cmap='gray')  # 显示图片
            plt.title(f'Pred: {predicted[i].item()}, True: {labels[i]}', fontsize=8)  # 默认标题

    plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 自动调整子图布局
    plt.show()  # 显示图像

#main

# test()
# for epoch in range(1, n_epochs + 1):
#   train(epoch)
#   test()

pic_show()
