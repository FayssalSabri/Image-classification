import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
def distance_matrix(X, Y):
    
    X_square = np.sum(X**2, axis=1).reshape(-1, 1)  # Shape (num_samples_x, 1)
    Y_square = np.sum(Y**2, axis=1).reshape(1, -1)  # Shape (1, num_samples_y)
    
    cross_term = np.dot(X, Y.T)  # Shape (num_samples_x, num_samples_y)
    
    dists = np.sqrt(X_square + Y_square - 2 * cross_term)
    
    return dists


def knn_predict(dists, labels_train, k):
    
    k_nearest_indices = np.argsort(dists, axis=1)[:, :k]  # Shape (num_test, k)
    
    k_nearest_labels = labels_train[k_nearest_indices]  # Shape (num_test, k)
    
    predicted_labels, _ = mode(k_nearest_labels, axis=1)
    predicted_labels = predicted_labels.flatten()  # Convert to 1D array
    
    return predicted_labels

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    
    dists = distance_matrix(data_test, data_train)
    
    predicted_labels = knn_predict(dists, labels_train, k)
    
    accuracy = np.mean(predicted_labels == labels_test)
    
    return accuracy

