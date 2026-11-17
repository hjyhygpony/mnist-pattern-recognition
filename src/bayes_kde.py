import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置随机种子保证结果可复现
np.random.seed(42)
torch.manual_seed(42)


# 1. 数据准备 (使用PyTorch的DataLoader)
def prepare_data(batch_size=60000):
    """准备MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST的均值和标准差
        transforms.Lambda(lambda x: x.view(-1))  # 展平图像
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # 使用大batch size一次性加载所有数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 获取numpy数组形式的数据
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    return X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()


# 2. 自定义核密度估计分类器
class KDEClassifier(BaseEstimator, ClassifierMixin):
    """基于核密度估计的贝叶斯分类器"""

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes_ = None
        self.kdes_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.kdes_ = []
        self.priors_ = []

        # 对每个类别训练一个KDE模型
        lb = LabelBinarizer()
        y_one_hot = lb.fit_transform(y)
        self.priors_ = np.mean(y_one_hot, axis=0)

        for i in range(len(self.classes_)):
            mask = (y == self.classes_[i])
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X[mask])
            self.kdes_.append(kde)
        return self

    def predict_proba(self, X):
        logprobs = np.array([kde.score_samples(X) for kde in self.kdes_]).T
        probs = np.exp(logprobs)
        probs = probs * self.priors_  # 乘以先验概率
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# 3. 绘制混淆矩阵
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


# 4. 实验主函数
def bayesian_experiment(X_train, X_test, y_train, y_test):
    # 检查并处理方差为0的特征
    variances = np.var(X_train, axis=0)
    zero_var_indices = np.where(variances == 0)[0]

    if len(zero_var_indices) > 0:
        print(f"Found {len(zero_var_indices)} features with zero variance. Applying PCA...")
        # 使用PCA降维
        pca = PCA(n_components=0.95)  # 保留95%的方差
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"Reduced dimensions: {X_train.shape[1]}")
    else:
        print("No zero-variance features found. Using original features.")

    # 定义要测试的维度列表
    dimensions = [10, 20, 30, 50, 100, X_train.shape[1]]
    dimensions = [d for d in dimensions if d <= X_train.shape[1]]

    # 存储结果
    results = {
        'gaussian': {'train_acc': [], 'test_acc': []},
        'kde': {'train_acc': [], 'test_acc': []}
    }

    for dim in dimensions:
        print(f"\nExperimenting with dimension: {dim}")

        # 如果当前维度小于特征数，使用PCA进一步降维
        if dim < X_train.shape[1]:
            pca = PCA(n_components=dim)
            X_train_dim = pca.fit_transform(X_train)
            X_test_dim = pca.transform(X_test)
        else:
            X_train_dim = X_train
            X_test_dim = X_test

        # 方法1: 高斯朴素贝叶斯
        print("Fitting Gaussian Naive Bayes...")
        gnb = GaussianNB()
        gnb.fit(X_train_dim, y_train)

        # 训练集评估
        train_pred = gnb.predict(X_train_dim)
        train_acc = accuracy_score(y_train, train_pred)
        results['gaussian']['train_acc'].append(train_acc)

        # 测试集评估
        test_pred = gnb.predict(X_test_dim)
        test_acc = accuracy_score(y_test, test_pred)
        results['gaussian']['test_acc'].append(test_acc)

        # 保存混淆矩阵
        plot_confusion_matrix(confusion_matrix(y_train, train_pred),
                              classes=[str(i) for i in range(10)],
                              title=f'GaussianNB Train Confusion Matrix (dim={dim})',
                              filename=f'bayesian_results/confusion_matrices/gaussian_train_cm_dim{dim}.png')

        plot_confusion_matrix(confusion_matrix(y_test, test_pred),
                              classes=[str(i) for i in range(10)],
                              title=f'GaussianNB Test Confusion Matrix (dim={dim})',
                              filename=f'bayesian_results/confusion_matrices/gaussian_test_cm_dim{dim}.png')

        print(f"GaussianNB - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # 方法2: 核密度估计
        print("Fitting Kernel Density Estimation...")
        kde = KDEClassifier(bandwidth=0.2)  # 较小的带宽适合MNIST数据
        kde.fit(X_train_dim, y_train)

        # 训练集评估
        train_pred = kde.predict(X_train_dim)
        train_acc = accuracy_score(y_train, train_pred)
        results['kde']['train_acc'].append(train_acc)

        # 测试集评估
        test_pred = kde.predict(X_test_dim)
        test_acc = accuracy_score(y_test, test_pred)
        results['kde']['test_acc'].append(test_acc)

        # 保存混淆矩阵
        plot_confusion_matrix(confusion_matrix(y_train, train_pred),
                              classes=[str(i) for i in range(10)],
                              title=f'KDE Train Confusion Matrix (dim={dim})',
                              filename=f'bayesian_results/confusion_matrices/kde_train_cm_dim{dim}.png')

        plot_confusion_matrix(confusion_matrix(y_test, test_pred),
                              classes=[str(i) for i in range(10)],
                              title=f'KDE Test Confusion Matrix (dim={dim})',
                              filename=f'bayesian_results/confusion_matrices/kde_test_cm_dim{dim}.png')

        print(f"KDE - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 绘制性能曲线
    plt.figure(figsize=(12, 6))
    plt.plot(dimensions, results['gaussian']['train_acc'], 'o-', label='Gaussian Train')
    plt.plot(dimensions, results['gaussian']['test_acc'], 'o-', label='Gaussian Test')
    plt.plot(dimensions, results['kde']['train_acc'], 's-', label='KDE Train')
    plt.plot(dimensions, results['kde']['test_acc'], 's-', label='KDE Test')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Accuracy')
    plt.title('Bayesian Classifier Performance vs Feature Dimension')
    plt.legend()
    plt.grid()
    plt.savefig('bayesian_results/performance/dimension_vs_accuracy.png')
    plt.close()

    return results, dimensions


# 5. 主函数
def main():
    # 创建保存结果的目录
    os.makedirs('bayesian_results/confusion_matrices', exist_ok=True)
    os.makedirs('bayesian_results/performance', exist_ok=True)

    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data()

    # 进行贝叶斯实验
    print("\nStarting Bayesian experiments...")
    results, dimensions = bayesian_experiment(X_train, X_test, y_train, y_test)

    # 打印最佳结果
    best_gaussian_idx = np.argmax(results['gaussian']['test_acc'])
    best_kde_idx = np.argmax(results['kde']['test_acc'])

    print("\nBest Results:")
    print(f"GaussianNB - Best dim: {dimensions[best_gaussian_idx]}, "
          f"Train Acc: {results['gaussian']['train_acc'][best_gaussian_idx]:.4f}, "
          f"Test Acc: {results['gaussian']['test_acc'][best_gaussian_idx]:.4f}")

    print(f"KDE - Best dim: {dimensions[best_kde_idx]}, "
          f"Train Acc: {results['kde']['train_acc'][best_kde_idx]:.4f}, "
          f"Test Acc: {results['kde']['test_acc'][best_kde_idx]:.4f}")


if __name__ == '__main__':
    main()