# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import scipy.io
import math

class SVM():
    def __init__(self,data, selected_features, positive_class, negative_class, kernel_type='linear', C=1, sigma_p=1):
        # data processing
        self.positive_data = selected_features[data['label'] == positive_class]
        self.negative_data = selected_features[data['label'] == negative_class]

        self.training_data = pd.concat([self.positive_data.head(25), self.negative_data.head(25)], axis=0)
        self.testing_data = pd.concat([self.positive_data.tail(25), self.negative_data.tail(25)], axis=0)

        self.x_train = self.training_data.values
        self.y_train = np.concatenate((np.ones(25), -np.ones(25)))

        self.x_test = self.testing_data.values
        self.y_test = np.concatenate((np.ones(25), -np.ones(25)))

        # parameters
        self.alpha = None
        self.C = C
        self.sigma = sigma_p
        self.p = sigma_p
        self.kernel_type = kernel_type
        self.b = None
        self.alpha_sum = None

    def kernel_function(self, x, y):
        if self.kernel_type == 'linear':
            k = x @ y
        elif self.kernel_type == 'rbf':
            norm_sq_neg = -1 * ( np.linalg.norm(x - y) ** 2 )
            k = math.exp(norm_sq_neg / (2 * (self.p ** 2)))
        elif self.kernel_type == 'polynomial':
            k = pow(x @ y, self.sigma)
        return k
    
    def solve_parameters(self):
        #  alpha parameters preprocessing
        n = len(self.x_train)
        lb = np.full(n, 0)
        ub = np.full(n, self.C)

        P = scipy.sparse.lil_matrix((n, n))
        for i in range(n):
            for j in range(n):
                k = self.kernel_function(self.x_train[i], self.x_train[j])
                P[i, j] = self.y_train[i] * self.y_train[j] * k

        P = P.tocsc()  # transfer into csc_matrix

        q = np.full(n, -1).T
        A = self.y_train
        A = scipy.sparse.csc_matrix(A)
        b = np.array([0])

        # use qpsolvers to solve alpha
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

        # handling alpha values
        eps = 2.2204e-16
        for i in range(alpha.size):
            if alpha[i] >= self.C - np.sqrt(eps):
                alpha[i] = self.C
                alpha[i] = np.round(alpha[i], 6)
            elif alpha[i] <= 0 + np.sqrt(eps):
                alpha[i] = 0
                alpha[i] = np.round(alpha[i], 6)
            else:
                alpha[i] = np.round(alpha[i], 6)
                #print(f"support vector: alpha = {alpha[i]}")

        print_alpha = np.round(alpha, 4)
        print('alpha = ', print_alpha)

        self.alpha = alpha
        self.alpha_sum = np.round(np.sum(self.alpha), 4)

        # solve b*
        sum = 0
        b_list = []

        for i in range(len(self.alpha)):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                sum = 0
                for j in range(len(self.alpha)):
                    k = self.kernel_function(self.x_train[j], self.x_train[i])
                    sum += self.alpha[j] * self.y_train[j] * k
                bias = 1.0 / self.y_train[i] - sum

                b_list.append(bias)
        
        self.b = np.mean(np.array(b_list))

        print('b = ', np.round(b_list, 4))


    def CR(self):
        predict = []
        for i in range(len(self.x_test)):
            sum_all = 0
            for j in range(len(self.x_train)):
                k = self.kernel_function(self.x_train[j], self.x_test[i])
                sum_all += (self.alpha[j]*self.y_train[j]*k)
            d = sum_all + self.b
            if d >= 0:
                predict.append(1)
            else:
                predict.append(-1)

        correct = np.sum(predict == self.y_test)
        accuracy = correct / len(self.x_test)
        print(f"Mode: {self.kernel_type} , Classification Rate (CR): {accuracy * 100:.2f}%")
        print('----------------------------------------------------------------')


    def auto_execute(cls, data, selected_features, positive_class, negative_class, kernel_type='linear', C=1, sigma_p=1):
        obj = cls(data, selected_features, positive_class, negative_class, kernel_type, C, sigma_p)
        obj.solve_parameters()
        obj.CR()


def main():
    data = pd.read_csv('C:\\Users\\willi\\Desktop\\ML\\HW2\\iris.txt', sep='\s+', header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})
    
    # initialize
    positive_class = 2
    negative_class = 3
    selected_features = data[['petal_length', 'petal_width']]

    # part 1
    # linear kernel-based svm with c = 1
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=1)
    # linear kernel-based svm with c = 10
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=10)
    # linear kernek-based svm with c = 100
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=100)

    # part 2
    # RBF kernel-based svm with C = 10, sigma = 5
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'rbf', C=10, sigma_p=5)
    # RBF kernel-based svm with C = 10, sigma = 1
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'rbf', C=10, sigma_p=1)
    # RBF kernel-based svm with C = 10, sigma = 0.5
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'rbf', C=10, sigma_p=0.5)
    # RBF kernel-based svm with C = 10, sigma = 0.1
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'rbf', C=10, sigma_p=0.1)
    # RBF kernel-based svm with C = 10, sigma = 0.05
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'rbf', C=10, sigma_p=0.05)

    # part 3
    # Polynomial kernel-based svm with C = 10, P = 1
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'polynomial', C=10, sigma_p=1)
    # Polynomial kernel-based svm with C = 10, P = 2
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'polynomial', C=10, sigma_p=2)
    # Polynomial kernel-based svm with C = 10, P = 3
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'polynomial', C=10, sigma_p=3)
    # Polynomial kernel-based svm with C = 10, P = 4
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'polynomial', C=10, sigma_p=4)
    # Polynomial kernel-based svm with C = 10, P = 5
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'polynomial', C=10, sigma_p=5)


if __name__ == '__main__':
    main()