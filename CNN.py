import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

"""
Normal:Normal  正常记录

DOS: back、land、neptune、pod、smurf、teardrop   拒绝服务攻击

Probing: ipsweep、nmap、portsweep、satan   监视和其他探测活动

R2L: ftp_write、guess_passwd、imap、multihop、phf、spy、warezclient、warezmaster   来自远程机器的非法访问

U2R: buffer overflow、loadmodule、perl、rootkit    普通用户对本地超级用户特权的非法访问
"""

# 定义要替换的类别映射关系
replace_map = {
    'back': 'DOS',
    'land': 'DOS',
    'neptune': 'DOS',
    'pod': 'DOS',
    'smurf': 'DOS',
    'teardrop': 'DOS',
    'ipsweep': 'Probing',
    'nmap': 'Probing',
    'portsweep': 'Probing',
    'satan': 'Probing',
    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R'
}


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 2, 1), nn.ReLU(),
            nn.Conv1d(32, 64, 2, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 2, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 2, 1), nn.ReLU(),
            nn.Conv1d(256, 512, 2, 1), nn.ReLU(),
            nn.Conv1d(512, 1024, 2, 1), nn.ReLU(),
            nn.Conv1d(1024, 1024, 2, 1), nn.ReLU(),
            nn.Conv1d(1024, 512, 2, 1), nn.ReLU(),
            nn.Conv1d(512, 512, 2, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Flatten(),
            nn.Linear(3 * 512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.Softmax(dim=1), nn.Dropout(0.1),
            nn.Linear(64, 5), nn.Softmax(dim=1)
        )

    def forward(self, the_x):
        return self.net(the_x)


def train_model(model, train_loader, epoch_nums, loss_func, optimizer, device):
    train_loss_all = []
    for epoch in range(epoch_nums):
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            model = model.to(device)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        print("train epoch %d, loss %s:" % (epoch + 1, train_loss))
    return train_loss_all


def test_model(the_model, the_model_path, the_test_loader):
    the_model.load_state_dict(torch.load(the_model_path))
    the_model.to('cuda:0')

    correct = 0
    total = 0

    with torch.no_grad():
        for (x, y) in the_test_loader:
            x, y = x.to('cuda:0'), y.to('cuda:0')
            pred = the_model(x)
            _, predicted = torch.max(pred.data, dim=1)
            correct += torch.sum((predicted == y))
            total += y.size(0)

    print(f'在测试集上的准确率为：{100 * correct / total}')


df1 = pd.read_csv('./NSL-KDD/KDDTrain+.txt', header=None)
df2 = pd.read_csv('./NSL-KDD/KDDTest+.txt', header=None)

# 将测试集中多余的标签删去
s1 = set(np.array(df1[41]).tolist())
df2 = df2[df2[41].isin(s1)]
print(len(df1))
print(len(df2))
df = pd.concat([df1, df2])

# 42列无用，删去
df.drop(df.columns[42], axis=1, inplace=True)

# 根据映射关系替换标签列中的类别
df[41] = df[41].replace(replace_map)

# 获取特征和标签
labels = df.iloc[:, 41]
df.drop(df.columns[41], axis=1, inplace=True)

# 删除不包含的特征
df.drop(df.columns[10:22 + 1], axis=1, inplace=True)
df.reset_index()

# 标签编码
le = LabelEncoder()
labels = le.fit_transform(labels).astype(np.int64)
res = {}
for cl in le.classes_:
    res.update({cl: le.transform([cl])[0]})
print(res)

# 重命名
data = df

# 特征编码
data[1] = le.fit_transform(data[1])
res = {}
for cl in le.classes_:
    res.update({cl: le.transform([cl])[0]})
print(res)
data[2] = le.fit_transform(data[2])
res = {}
for cl in le.classes_:
    res.update({cl: le.transform([cl])[0]})
print(res)
data[3] = le.fit_transform(data[3])
res = {}
for cl in le.classes_:
    res.update({cl: le.transform([cl])[0]})
print(res)

data = df.values
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# 归一化处理，分别对训练集和测试集进行归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 转换成 Tensor 类型
# x_train = torch.from_numpy(x_train).float().unsqueeze(1)
# x_test = torch.from_numpy(x_test).float().unsqueeze(1)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()

# 创建对应的 TensorDataset 和 DataLoader
train_dataset = TensorDataset(x_train, torch.from_numpy(y_train))
test_dataset = TensorDataset(x_test, torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = CNN()

num_epochs = 100
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loss_all = train_model(model, train_loader, num_epochs, criterion, optimizer, device)
torch.save(model.state_dict(), 'model.pth')
test_model(model, 'model.pth', test_loader)
