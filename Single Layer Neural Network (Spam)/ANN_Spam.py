
import sklearn.metrics as metrics
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

def load_dataset():
    data = scipy.io.loadmat('spam.mat')
    X_test = data['Xtest']
    X_train = data['Xtrain']
    y_train = data['ytrain']
    return X_train, y_train, X_test

def sigmoid(x):
    "Numerically-stable sigmoid function."
    return 1 / (1 + np.exp(-x))

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    error_lst = []
    plotFreq = 1 #plot every plotFreq computations
    beta = np.random.random((X_train.shape[1], 1))
    for i in range(num_iter):
        mu = sigmoid(X_train @ beta)
        update = (1/X_train.shape[0])*alpha*(2*reg*beta - (X_train.T @ (y_train - mu)))
        beta = beta - update
        if i % plotFreq == 0:
            summation = y_train.T @ np.log(mu) + (1-y_train).T @ np.log(1 - mu)
            error_lst.append(reg*np.linalg.norm(beta) - summation)
    error_arr = np.array(error_lst)[:,0,0]
    plt.plot(np.arange(len(error_arr))*plotFreq, error_arr)
    plt.xlabel("# Iterations")
    plt.ylabel("Training Error")
    plt.title("Batch Gradient Descent Training Error through Time")
    plt.show()    
    return beta

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    beta = np.random.random((X_train.shape[1], 1))
    error_lst = []
    plotFreq = 1 #plot every 100 computations
    count = 0
    for i in range(num_iter // X_train.shape[0]):
        order = np.arange(X_train.shape[0])
        np.random.shuffle(order)
        for i in order:
            count += 1
            x = X_train[i].reshape((X_train.shape[1], 1)) #d x 1
            output = beta.T @ X_train[i] #1 x d x d x 1 = 1 x 1
            mu = sigmoid(output) #1 x 1
            difference = y_train[i] - mu #1 x 1
            mult = difference*x #d x 1
            beta = beta - alpha*(2*reg*beta - mult)
            if i % plotFreq == 0:
                mu = sigmoid(X_train @ beta)
                summation = y_train.T @ np.log(mu) + (1-y_train).T @ np.log(1 - mu)
                error_lst.append(reg*np.linalg.norm(beta) - summation)
    error_arr = np.array(error_lst)[:,0,0]
    plt.plot(np.arange(len(error_arr))*plotFreq, error_arr)
    plt.xlabel("# Iterations")
    plt.ylabel("Training Error")
    plt.title("Stochastic Gradient Descent Training Error through Time")
    plt.show()    
    return beta

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    output = X @ model
    mu = sigmoid(output)
    for i in range(len(mu)):
        if mu[i] > 0.5:
            mu[i] = 1
        else:
            mu[i] = 0
    return mu

def standardize1(X_train):
    """Standardize the columns to have mean 0 and unit variance"""
    for i in range(X_train.shape[1]):
        column = X_train[:,i]
        mean = np.mean(column)
        std = np.std(column)
        X_train[:,i] = (column - mean)/std
    return X_train

def standardize2(X_train):
    """Transform the features using log(x_ij + 0.1)"""
    return np.log(X_train + 0.1)

def standardize3(X_train):
    """Binarize the features using I(x_ij > 0)"""
    return np.where(X_train > 0, 1, 0)

if __name__ == "__main__":
    X_train, y_train, X_test = load_dataset()

    X_train = standardize2(X_train) #choose from standardize1, standardize2, standardize3
    X_test = standardize2(X_test)

    model = train_gd(X_train, y_train, reg=0, alpha=0.1, num_iter=60000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))

    model = train_sgd(X_train, y_train, reg=0.25, alpha=1, num_iter=60000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))
    
    ##CONTEST###
    # import csv

    # f = open("predictions_nophi.csv", 'wt')
    # try:
    #     writer = csv.writer(f)
    #     writer.writerow( ('Id', 'Category') )
    #     for i in range(len(pred_labels_test)):
    #         writer.writerow( (i+1, int(pred_labels_test[i][0])) )
    # finally:
    #     f.close()
