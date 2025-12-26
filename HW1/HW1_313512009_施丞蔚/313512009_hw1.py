import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

# 讀取數據集，使用 np.genfromtxt 以處理不規則空格
data = np.genfromtxt('C:\\Users\\willi\\Desktop\\ML\\HW1\\iris.txt', delimiter=None)

# 提取特徵和標籤
X = data[:, :4]  # 前四列為特徵
y = data[:, 4]   # 最後一列為標籤

# 特徵名稱
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

# 繪製特徵兩兩組合的散佈圖
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, (x_idx, y_idx) in enumerate(pairs):
    plt.figure(i)
    for label in np.unique(y):
        plt.scatter(X[y == label, x_idx], X[y == label, y_idx], label=f"Class {int(label)}")
    plt.xlabel(features[x_idx])
    plt.ylabel(features[y_idx])
    plt.title(f'{features[x_idx]} vs {features[y_idx]}')
    plt.legend()
    plt.show()

# K-NN 分類器
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn(X_train, y_train, X_test, k=1):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_neighbors = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_neighbors]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

# 按類別劃分數據
def split_data(X, y):
    unique_classes = np.unique(y)
    train_data = []
    test_data = []

    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        mid_point = len(class_indices) // 2
        train_indices = class_indices[:mid_point]
        test_indices = class_indices[mid_point:]

        train_data.append((X[train_indices], y[train_indices]))
        test_data.append((X[test_indices], y[test_indices]))

    return train_data, test_data

# 生成所有可能的特徵組合
def generate_feature_combinations(X):
    combinations_list = []
    for r in range(1, 5):  # r 為特徵的選擇數量 (1, 2, 3, 4)
        combinations_list.extend(combinations(range(X.shape[1]), r))
    return combinations_list

# 計算分類率
def evaluate_knn(X, y, k=1):
    classification_rates = []
    feature_combinations = generate_feature_combinations(X)

    for comb in feature_combinations:
        X_subset = X[:, comb]
        
        # 按類別劃分數據
        train_data, test_data = split_data(X_subset, y)
        
        # Combine training and testing data
        X_train = np.vstack([td[0] for td in train_data])
        y_train = np.hstack([td[1] for td in train_data])
        X_test = np.vstack([td[0] for td in test_data])
        y_test = np.hstack([td[1] for td in test_data])
        
        # 計算第一個分類率
        y_pred = knn(X_train, y_train, X_test, k=k)
        accuracy1 = np.mean(y_pred == y_test)
        
        # 互換訓練和測試集
        y_pred_swap = knn(X_test, y_test, X_train, k=k)
        accuracy2 = np.mean(y_pred_swap == y_train)
        
        # 計算平均分類率
        avg_accuracy = (accuracy1 + accuracy2) / 2
        classification_rates.append(avg_accuracy * 100)  # 轉換為百分比

    return classification_rates, feature_combinations

# 對所有可能的特徵組合計算分類率 (K=1 和 K=3)
k1_rates, k1_combinations = evaluate_knn(X, y, k=1)
k3_rates, k3_combinations = evaluate_knn(X, y, k=3)

# 建立表格資料並顯示結果
results = pd.DataFrame({
    'Features': [str(comb) for comb in k1_combinations],
    'K=1 Classification Rate (%)': [round(rate, 2) for rate in k1_rates],
    'K=3 Classification Rate (%)': [round(rate, 2) for rate in k3_rates]
})

print(results)
