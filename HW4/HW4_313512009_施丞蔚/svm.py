import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import scipy.io
import math
from collections import defaultdict

class SVM():
    def __init__(self,x_train, y_train, x_test, y_test, kernel_type='linear', C=1, sigma_p=1):
        # data processing
        #self.positive_data = selected_features[data['label'] == positive_class]
        #self.negative_data = selected_features[data['label'] == negative_class]

        #self.training_data = pd.concat([self.positive_data.head(25), self.negative_data.head(25)], axis=0)
        #self.testing_data = pd.concat([self.positive_data.tail(25), self.negative_data.tail(25)], axis=0)

        #self.x_train = self.training_data.values
        #self.y_train = np.concatenate((np.ones(25), -np.ones(25)))

        #self.x_test = x_test
        #self.y_test = np.concatenate((np.ones(25), -np.ones(25)))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # parameters
        self.alpha = None
        self.C = C
        self.sigma = sigma_p
        self.p = sigma_p
        self.kernel_type = kernel_type
        self.b = None
        self.alpha_sum = None

    def kernel_function(self, x, y):
        k = 0
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
        

        self.alpha = alpha
        self.alpha_sum = np.round(np.sum(self.alpha), 4)
        #print('alpha_sum = ', self.alpha_sum)

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

        #print('b = ', np.round(b_list, 4))


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

        return predict


    def auto_execute(cls, x_train, y_train, x_test, y_test, kernel_type='linear', C=1, sigma_p=1):
        obj = cls(x_train, y_train, x_test, y_test, kernel_type, C, sigma_p)
        obj.solve_parameters()

        predict = obj.CR()

        return predict


def vote(list_1, list_2, list_3, real):
    correct_prediction = 0
    
    for i in range(len(list_1)):
        counts = defaultdict(int)  # Create a dictionary to count occurrences
        counts[list_1[i]] += 1
        counts[list_2[i]] += 1
        counts[list_3[i]] += 1

        max_count = max(counts.values())  # Find the maximum count
        predicted_values = [key for key, value in counts.items() if value == max_count]

        if len(predicted_values) == 1:
            predicted_value = predicted_values[0]

            if real[i] == predicted_value:  # Compare the entire array, not individual elements
                correct_prediction += 1
        else:
            return 0  # If there is a tie in the voting, return 0
    
    if round(correct_prediction / len(real), 4) < 0.35:
        return 0
    else:
        return round(correct_prediction / len(real), 4)



def main():
    data = pd.read_csv('C:\\Users\\willi\\Desktop\\ML\\HW4\\iris.txt', sep='\s+', header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})
    
    c_values = []
    sigma_values = []
    accuracy_values = []
    accuracy_values_1 = []
    accuracy_values_2 = []

    best_accuracy = 0  
    best_c = None
    best_sigma = None

    # search range
    C_list = [1, 5, 10, 50, 100, 500, 1000]
    sigma_list = [1.05**(-i) for i in range(100, -100, -5)]
    

    selected_features = data[['sepal_length','sepal_width','petal_length', 'petal_width']]

    real = [i for i in range(1, 4) for _ in range(25)]

    #print(real)

    for c in C_list:
        for sigma in sigma_list:
    
            # fold 1
            # testing_data
            selected_data = data.groupby('label').apply(lambda x: x.tail(25)).reset_index(drop=True)
            selected_data = selected_data.drop(columns=['label'])
            x_test = selected_data.to_numpy()

            # 1 vs 2
            positive_class = 1
            negative_class = 2

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_1 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_1)):
                if predict_1[i] == 1:
                    predict_1[i] = 1
                else:
                    predict_1[i] = 2

            # 2 vs 3
            positive_class = 2
            negative_class = 3

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_2 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_2)):
                if predict_2[i] == 1:
                    predict_2[i] = 2
                else:
                    predict_2[i] = 3

            # 1 vs 3
            positive_class = 1
            negative_class = 3

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_3 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_3)):
                if predict_3[i] == 1:
                    predict_3[i] = 1
                else:
                    predict_3[i] = 3

            accuracy_fold1 = vote(predict_1, predict_2, predict_3, real)
            #print(accuracy_fold1*100)

            accuracy_values_1.append(accuracy_fold1)

            # fold 2
            # testing_data
            selected_data = data.groupby('label').apply(lambda x: x.head(25)).reset_index(drop=True)
            selected_data = selected_data.drop(columns=['label'])
            x_test = selected_data.to_numpy()

            # 1 vs 2
            positive_class = 1
            negative_class = 2

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_1 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_1)):
                if predict_1[i] == 1:
                    predict_1[i] = 1
                else:
                    predict_1[i] = 2

            # 2 vs 3
            positive_class = 2
            negative_class = 3

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_2 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_2)):
                if predict_2[i] == 1:
                    predict_2[i] = 2
                else:
                    predict_2[i] = 3

            # 1 vs 3
            positive_class = 1
            negative_class = 3

            positive_data = selected_features[data['label'] == positive_class]
            negative_data = selected_features[data['label'] == negative_class]

            training_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))
            y_test = np.concatenate((np.ones(25), -np.ones(25)))

            predict_3 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)
            for i in range(len(predict_3)):
                if predict_3[i] == 1:
                    predict_3[i] = 1
                else:
                    predict_3[i] = 3

            accuracy_fold2 = vote(predict_1, predict_2, predict_3, real)
            #print(accuracy_fold2*100)

            accuracy_values_2.append(accuracy_fold2)

            accuracy = (accuracy_fold1 + accuracy_fold2) / 2

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = c
                best_sigma = round(sigma, 4)

            c_values.append(c)
            sigma_values.append(round(sigma, 4))
            accuracy_values.append(f'{accuracy:.2%}')
    
    print(f"Best CR: {best_accuracy:.2%}")
    print(f"c: {best_c}")
    print(f"sigma: {best_sigma}")

    print("fold 1 accuracy = ", max(accuracy_values_1)*100)
    print("fold 2 accuracy = ", max(accuracy_values_2)*100)

    result_df = pd.DataFrame({
        'C': c_values,
        'sigma': sigma_values,
        'Accuracy': accuracy_values
    })

    result_df.set_index(['C', 'sigma'], inplace=True)
    result_df.sort_index(inplace=True)

    result_pivot = result_df.pivot_table(values='Accuracy', index='sigma', columns='C', aggfunc='first')

    print(result_pivot)



if __name__ == '__main__':
    main()