import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
from itertools import combinations

class LDA:
    def __init__(self):
        self.mean_class_1 = None
        self.mean_class_2 = None
        self.w_T = None
        self.b = None
        self.TPR = None
        self.TNR = None
        self.predicted_list = None

    def fit(self, X, y, C=1):
        class_1 = X[y == 1]
        class_2 = X[y == 0]

        n1 = class_1.shape[0]
        n2 = class_2.shape[0]

        p1 = n1 / (n1 + n2)
        p2 = n2 / (n1 + n2)

        self.mean_class_1 = np.mean(class_1, axis=0)
        self.mean_class_2 = np.mean(class_2, axis=0)

        covariance_1 = np.cov(class_1, rowvar=False)
        covariance_2 = np.cov(class_2, rowvar=False)
        covariance = covariance_1 * p1 + covariance_2 * p2

        # 確保 covariance 是 2 維的，即使只有一個特徵
        if covariance.ndim == 0:
            covariance = np.array([[covariance]])
        elif covariance.ndim == 1:
            covariance = covariance.reshape(1, -1)

        # 使用偽逆矩陣以提高數值穩定性
        self.w_T = (self.mean_class_1 - self.mean_class_2).T @ np.linalg.pinv(covariance)
        self.b = -0.5 * self.w_T @ (self.mean_class_1 + self.mean_class_2) + np.log(C * (p1 / p2))

        # 四捨五入至小數點後兩位
        self.w_T = np.round(self.w_T, 2)
        self.b = round(self.b, 2)

        return self.w_T, self.b

    def decision_function(self, X):
        scores = X @ self.w_T + self.b
        return scores

    def predict(self, X, threshold=0):
        scores = self.decision_function(X)
        predicted = (scores > threshold).astype(int)
        self.predicted_list = predicted.tolist()
        return predicted

    def compute_balanced_accuracy(self, X, y_true):
        predictions = self.predict(X)
        TP = np.sum((predictions == 1) & (y_true == 1))
        TN = np.sum((predictions == 0) & (y_true == 0))
        FP = np.sum((predictions == 1) & (y_true == 0))
        FN = np.sum((predictions == 0) & (y_true == 1))

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0

        balanced_accuracy = (TPR + TNR) / 2
        self.TPR = TPR
        self.TNR = TNR

        return balanced_accuracy * 100  # 轉換為百分比

def sequential_forward_selection(X, y):
    print("------------------------Sequential Forward Selection (SFS)-----------------------------")
    num_features = X.shape[1]
    selected_features = []
    best_balanced_accuracy = 0
    best_feature_set = []

    for i in range(num_features):
        remaining_features = [f for f in range(num_features) if f not in selected_features]
        local_best_feature = None
        local_best_accuracy = 0

        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]

            # 2-Fold Cross Validation
            fold1_X, fold2_X, fold1_y, fold2_y = train_test_split(
                X_subset, y, test_size=0.5, random_state=42
            )

            lda = LDA()
            lda.fit(fold1_X, fold1_y)
            acc1 = lda.compute_balanced_accuracy(fold2_X, fold2_y)

            lda.fit(fold2_X, fold2_y)
            acc2 = lda.compute_balanced_accuracy(fold1_X, fold1_y)

            average_accuracy = (acc1 + acc2) / 2

            if average_accuracy > local_best_accuracy:
                local_best_accuracy = average_accuracy
                local_best_feature = feature

        if local_best_feature is not None:
            selected_features.append(local_best_feature)
            print(f"Step {i + 1}: Selected Feature {local_best_feature}, Balanced Accuracy: {local_best_accuracy:.2f}%")

            if local_best_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = local_best_accuracy
                best_feature_set = selected_features.copy()

    print(" [-] Best Balanced Accuracy:", round(best_balanced_accuracy, 2), "%")
    print(" [-] Optimal Feature Subset (indices):", best_feature_set)
    return best_feature_set, best_balanced_accuracy

def fisher_criterion(X, y):
    print("------------------------Fisher’s Criterion-----------------------------")
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Fisher’s Criterion is implemented for binary classification only.")

    class_1 = X[y == classes[0]]
    class_2 = X[y == classes[1]]

    mean_class_1 = np.mean(class_1, axis=0)
    mean_class_2 = np.mean(class_2, axis=0)

    # Within-class scatter matrix
    Sw = np.cov(class_1, rowvar=False) + np.cov(class_2, rowvar=False)

    # Between-class scatter matrix
    mean_diff = (mean_class_1 - mean_class_2).reshape(-1, 1)
    Sb = mean_diff @ mean_diff.T

    # Fisher's score for each feature
    fisher_scores = []
    for i in range(X.shape[1]):
        numerator = Sb[i, i]
        denominator = Sw[i, i] if Sw[i, i] != 0 else 1e-6  # 避免除以零
        fisher_score = numerator / denominator
        fisher_scores.append(fisher_score)

    fisher_scores = np.array(fisher_scores)
    sorted_indices = np.argsort(fisher_scores)[::-1]

    return sorted_indices, fisher_scores[sorted_indices]

def fisher_feature_selection(X, y):
    sorted_indices, fisher_scores = fisher_criterion(X, y)
    print("Fisher’s scores (sorted):", fisher_scores)

    best_balanced_accuracy = 0
    best_feature_set = []
    optimal_N = 0

    for N in range(1, X.shape[1] + 1):
        top_N_features = sorted_indices[:N]
        X_subset = X[:, top_N_features]

        # 2-Fold Cross Validation
        fold1_X, fold2_X, fold1_y, fold2_y = train_test_split(
            X_subset, y, test_size=0.5, random_state=42
        )

        lda = LDA()
        lda.fit(fold1_X, fold1_y)
        acc1 = lda.compute_balanced_accuracy(fold2_X, fold2_y)

        lda.fit(fold2_X, fold2_y)
        acc2 = lda.compute_balanced_accuracy(fold1_X, fold1_y)

        average_accuracy = (acc1 + acc2) / 2

        print(f"Top-{N} features: {top_N_features}, Balanced Accuracy: {average_accuracy:.2f}%")

        if average_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = average_accuracy
            best_feature_set = top_N_features.copy()
            optimal_N = N

    print(" [-] Best Balanced Accuracy:", round(best_balanced_accuracy, 2), "%")
    print(" [-] Optimal Feature Subset (indices):", best_feature_set)
    print(" [-] Number of Features:", optimal_N)
    return best_feature_set, best_balanced_accuracy

def plot_feature_selection_results(sfs_results, fisher_results):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sfs_results)+1), sfs_results, label='SFS Balanced Accuracy', marker='o')
    plt.plot(range(1, len(fisher_results)+1), fisher_results, label='Fisher’s Criterion Balanced Accuracy', marker='s')
    plt.xlabel('Number of Features')
    plt.ylabel('Balanced Accuracy (%)')
    plt.title('Feature Selection Methods Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 載入乳癌資料集
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target  # 0 = malignant, 1 = benign

    # 將標籤轉換為 0 和 1，其中 1 為良性 (benign), 0 為惡性 (malignant)
    y = y  # 已經是 0 和 1

    print("========================================")
    print("Part1: Sequential Forward Selection (SFS)")
    optimal_features_sfs, best_accuracy_sfs = sequential_forward_selection(X, y)
    print("========================================\n")

    print("========================================")
    print("Part2: Fisher’s Criterion")
    optimal_features_fisher, best_accuracy_fisher = fisher_feature_selection(X, y)
    print("========================================\n")

if __name__ == '__main__':
    main()
