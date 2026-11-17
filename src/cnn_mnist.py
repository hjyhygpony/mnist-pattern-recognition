import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import seaborn as sns

# 设置随机种子保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据准备
def prepare_data(batch_size=64):
    """准备MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 3. 绘制混淆矩阵（增强版）
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, filename=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                cbar=True, annot_kws={'size': 12})
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 4. 训练和评估函数（完整版）
def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录指标
    train_losses = []
    train_accs = []
    test_accs = []

    # 创建结果目录
    os.makedirs('results/performance_curves', exist_ok=True)
    os.makedirs('results/confusion_matrices', exist_ok=True)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        train_true = []
        train_pred = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predicted.cpu().numpy())

        # 计算训练集指标
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试阶段
        model.eval()
        test_true = []
        test_pred = []
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                test_true.extend(labels.cpu().numpy())
                test_pred.extend(predicted.cpu().numpy())

        test_acc = test_correct / test_total
        test_accs.append(test_acc)

        print(f'Epoch [{epoch + 1}/{epochs}] | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Test Acc: {test_acc:.4f}')

    # 绘制性能曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('results/performance_curves/performance.png')
    plt.close()

    # 生成训练集混淆矩阵
    train_cm = confusion_matrix(train_true, train_pred)
    plot_confusion_matrix(train_cm, classes=[str(i) for i in range(10)],
                          title='Train Confusion Matrix',
                          filename='results/confusion_matrices/train_cm.png')

    # 生成测试集混淆矩阵
    test_cm = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(test_cm, classes=[str(i) for i in range(10)],
                          title='Test Confusion Matrix',
                          filename='results/confusion_matrices/test_cm.png')

    # 保存分类报告
    with open('results/classification_report.txt', 'w') as f:
        f.write("=== Training Set Classification Report ===\n")
        f.write(classification_report(train_true, train_pred, target_names=[str(i) for i in range(10)]))
        f.write("\n=== Test Set Classification Report ===\n")
        f.write(classification_report(test_true, test_pred, target_names=[str(i) for i in range(10)]))

    return {
        'train_acc': train_accs[-1],
        'test_acc': test_accs[-1],
        'train_cm': train_cm,
        'test_cm': test_cm
    }


# 5. 主函数
def main():
    # 准备数据
    train_loader, test_loader = prepare_data()

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN()

    # 训练和评估
    results = train_and_evaluate(model, train_loader, test_loader,
                                 epochs=15, lr=0.001, device=device)

    # 打印最终结果
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {results['train_acc']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print("Confusion matrices and reports saved in 'results' directory")


if __name__ == '__main__':
    main()