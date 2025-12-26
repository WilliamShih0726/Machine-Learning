import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from collections import defaultdict


class LDA():
    def __init__(self):
        self.mean_class_1 = None
        self.mean_class_2 = None
        self.w_T = None
        self.b = None
        self.TPR = None
        self.FPR = None
        self.predicted_list = None


    def fit(self, x, y, C=1):
        class_1 = x[y == 1]
        class_2 = x[y == 0]

        n1 = class_1.shape[0]
        n2 = class_2.shape[0]
        
        p1 = n1/(n1+n2)
        p2 = n2/(n1+n2)

        self.mean_class_1 = np.mean(class_1, axis = 0)
        self.mean_class_2 = np.mean(class_2, axis = 0)

        covariance_1 = np.cov(class_1, rowvar=False)
        covariance_2 = np.cov(class_2, rowvar=False)
        covariance = covariance_1*p1 + covariance_2*p2

        self.w_T = (self.mean_class_1 - self.mean_class_2).T @ np.linalg.inv(covariance)
        self.b = -0.5 * self.w_T @ (self.mean_class_1 + self.mean_class_2) - np.log( C*(p2 / p1) )

        self.w_T = np.round(self.w_T, 2)
        self.b = round(self.b, 2)

        return self.w_T, self.b
        

    def LDA_decision_function(self, x, y_true):
        TP = FP = FN = TN = 0
        x = np.array(x)
        self.predicted_list = []

        for i in range(len(x)):
            x_col = x[i].T

            g = self.w_T @ x_col + self.b
            predicted_class = 1 if g > 0 else 0
            self.predicted_list.append(predicted_class)

            if predicted_class == y_true[i] and predicted_class == 1:
                TP += 1
            if predicted_class == y_true[i] and predicted_class == 0:
                TN += 1
            if predicted_class != y_true[i] and predicted_class == 1:
                FP += 1
            if predicted_class != y_true[i] and predicted_class == 0:
                FN += 1

        self.TPR = TP / (TP + FN)
        self.FPR = FP / (FP + TN)

        accuracy = (TP+TN) / len(y_true)

        return accuracy
    
    def return_TPR_AND_FPR(self):
        return self.TPR, self.FPR
    
    def return_predicted_list(self):
        return self.predicted_list


def two_fold_cross_variation_lda(data, selected_features, positive_class, negative_class, C=1):
    # adopt the third and forth typees of features
    positive_data = selected_features[data['label'] == positive_class]
    negative_data = selected_features[data['label'] == negative_class]

    # split the data into training set & testing set
    training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)
    testing_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    lda = LDA()

    training_w, training_b = lda.fit(x_train, y_train, C)
    testing_accuracy = (lda.LDA_decision_function(x_test, y_test))*100

    testing_w, testing_b = lda.fit(x_test, y_test, C)
    training_accuracy = (lda.LDA_decision_function(x_train, y_train))*100

    average_accuracy = (testing_accuracy + training_accuracy) / 2

    print('training w = ', training_w, 'training b = ', training_b,'testing accuracy', testing_accuracy, '%')
    print('testing w = ', testing_w, 'testing b = ', testing_b,'training accuracy', training_accuracy, '%')
    print('average accuracy = ', average_accuracy, '%')


def ROC_AND_AUC(data, selected_features, positive_class, negative_class):
    # adopt the third and forth typees of features
    positive_data = selected_features[data['label'] == positive_class]
    negative_data = selected_features[data['label'] == negative_class]

    # split the data into training set & testing set
    training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)
    testing_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    FPR_X = []
    TPR_Y = []

    for exp in range(-1000, 1000, 1): 
        C = 10 ** (exp / 10.0)

        lda = LDA()
        lda.fit(x_train, y_train, C)
        lda.LDA_decision_function(x_test, y_test)
        TPR_test, FPR_test = lda.return_TPR_AND_FPR()

        lda.fit(x_test, y_test, C)
        lda.LDA_decision_function(x_train, y_train)
        TPR_train, FPR_train = lda.return_TPR_AND_FPR()

        TPR_Y.append((TPR_test + TPR_train)/2)
        FPR_X.append((FPR_test + FPR_train)/2)

        # print((TPR_test + TPR_train)/2, (FPR_test + FPR_train)/2)

    roc_auc = auc(FPR_X, TPR_Y)

    print("AUC = ", roc_auc)

    # Plot the ROC curve
    plt.plot(FPR_X, TPR_Y, linestyle='--')
    plt.xlabel('False Positive Rate (FPR_X)')
    plt.ylabel('True Positive Rate (TPR_Y)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()


def vote(list_1, list_2, list_3, real):
    correct_prediction = 0
    
    for i in range(len(list_1)):
        counts = defaultdict(int)  # Create a dictionary to count occurrences
        counts[list_1[i]] += 1
        counts[list_2[i]] += 1
        counts[list_3[i]] += 1

        max_count = max(counts.values())  # Find the maximum count
        predicted_value = [key for key, value in counts.items() if value == max_count][0]

        if real[i] == predicted_value:  # Compare the entire array, not individual elements
            correct_prediction += 1
    
    return round(correct_prediction / len(real), 4)


def one_against_one_strategy(data):
    selected_features = data[['petal_length', 'petal_width']]

    class_1 = 1
    class_2 = 2
    class_3 = 3

    data_1 = selected_features[data['label'] == class_1]
    data_2 = selected_features[data['label'] == class_2]
    data_3 = selected_features[data['label'] == class_3]

    # fold-1
    # class_1 v.s. class_2 - (1)
    training_data = pd.concat([data_1.head(25), data_2.head(25)], axis=0)
    testing_data = pd.concat([data_1.tail(25), data_2.tail(25), data_3.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(50)))

    lda_1_2 = LDA()

    lda_1_2.fit(x_train, y_train)
    lda_1_2.LDA_decision_function(x_test, y_test)
    predicted_1 = lda_1_2.return_predicted_list()
    predicted_1 = [1 if x == 1 else 2 for x in predicted_1]
 
    # class_1 v.s. class_3 - (1)
    training_data = pd.concat([data_1.head(25), data_3.head(25)], axis=0)
    testing_data = pd.concat([data_1.tail(25), data_2.tail(25), data_3.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(50)))

    lda_1_3 = LDA()
    

    lda_1_3.fit(x_train, y_train)
    lda_1_3.LDA_decision_function(x_test, y_test)
    predicted_2 = lda_1_3.return_predicted_list()
    predicted_2 = [1 if x == 1 else 3 for x in predicted_2]

    # class_2 v.s. class_3 (1)
    training_data = pd.concat([data_2.head(25), data_3.head(25)], axis=0)
    testing_data = pd.concat([data_1.tail(25), data_2.tail(25), data_3.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(50)))

    lda_2_3 = LDA()

    lda_2_3.fit(x_train, y_train)
    lda_2_3.LDA_decision_function(x_test, y_test)
    predicted_3 = lda_2_3.return_predicted_list()
    predicted_3 = [2 if x == 1 else 3 for x in predicted_3]

    real = np.concatenate((np.ones(25), np.full(25, 2), np.full(25, 3)))

    accuracy_1 = vote(predicted_1, predicted_2, predicted_3, real)
    print('fold-1 = ', 100*accuracy_1, '%')

    # fold-2
    # class_1 v.s. class_2 - (2)
    training_data = pd.concat([data_1.head(25), data_2.head(25), data_3.head(25)], axis=0)
    testing_data = pd.concat([data_1.tail(25), data_2.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(50)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    lda_1_2 = LDA()

    lda_1_2.fit(x_test, y_test)
    lda_1_2.LDA_decision_function(x_train, y_train)
    predicted_1 = lda_1_2.return_predicted_list()
    predicted_1 = [1 if x == 1 else 2 for x in predicted_1]
 
    # class_1 v.s. class_3 - (1)
    training_data = pd.concat([data_1.head(25), data_2.head(25), data_3.head(25)], axis=0)
    testing_data = pd.concat([data_1.tail(25), data_3.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(50)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    lda_1_3 = LDA()

    lda_1_3.fit(x_test, y_test)
    lda_1_3.LDA_decision_function(x_train, y_train)
    predicted_2 = lda_1_3.return_predicted_list()
    predicted_2 = [1 if x == 1 else 3 for x in predicted_2]

    # class_2 v.s. class_3 (1)
    training_data = pd.concat([data_1.head(25), data_2.head(25), data_3.head(25)], axis=0)
    testing_data = pd.concat([data_2.tail(25), data_3.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(50)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    lda_2_3 = LDA()

    lda_2_3.fit(x_test, y_test)
    lda_2_3.LDA_decision_function(x_train, y_train)
    predicted_3 = lda_2_3.return_predicted_list()
    predicted_3 = [2 if x == 1 else 3 for x in predicted_3]

    real = np.concatenate((np.ones(25), np.full(25, 2), np.full(25, 3)))

    accuracy_2 = vote(predicted_1, predicted_2, predicted_3, real)
    print('fold-2 = ', 100*accuracy_2, '%')

    print('average = ',((accuracy_1 + accuracy_2)/2)*100, "%")


def main():
    data = pd.read_csv('C:\\Users\\willi\\Desktop\\ML\\HW2\\iris.txt', sep='\s+', header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})

    # Q1 : Use 2-fold cross-validation to calculate the CR value with LDA Classifier
    print('Q1')
    # positive class = Versicolor  /  negative class = Virginica
    positive_class = 2
    negative_class = 3

    C = 1

    selected_features = data[['petal_length', 'petal_width']]
    two_fold_cross_variation_lda(data, selected_features, positive_class, negative_class, C)
    print('--------------------------------')

    # Q2 : Plot the ROC and calculate the AUC scores
    print('Q2')
    # positive class = Virginica  /  negative class = Versicolor
    positive_class = 3
    negative_class = 2

    print('Feature = sepal_length, sepal_width, petal_length, petal_width')
    # Used all features
    selected_features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)

    # Use sepal_length & sepal_width
    print('Feature = sepal_length, sepal_width')
    selected_features = data[['sepal_length', 'sepal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)

    # Use petal_length & petal_width
    print('Feature = petal_length, petal_width')
    selected_features = data[['petal_length', 'petal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)
    print('--------------------------------')

    # Q3 : One against one strategy
    print('Q3')
    one_against_one_strategy(data)


if __name__ == '__main__':
    main()