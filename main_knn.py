import numpy as np
import matplotlib.pyplot as plt
import os

from read_cifar import read_cifar, split_dataset,read_cifar_batch
from knn import evaluate_knn

def plot_knn_accuracy_vs_k(data, labels, split_factor=0.9, k_range=range(1, 21)):

    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split_factor)
    accuracies = []
    for k in k_range:
        accuracy = evaluate_knn(data_train, labels_train, data_test, labels_test, k)
        accuracies.append(accuracy)
        print(f"Accuracy for k={k}: {accuracy:.4f}")
        
    plt.figure()
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs. Number of Neighbors (k)")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/knn.png")
    plt.show()


if __name__ == "__main__":
    batch_file_path = r"C:\Users\LENOVO\OneDrive\Desktop\Master\Deep learning\TD1-Image-classification\image-classification\data\cifar-10-python\cifar-10-batches-py\data_batch_1"
    
    
    data, labels = read_cifar_batch(batch_file_path)
    
    print("Question 1 :")
    print("Data shape:", data.shape)  
    print("Labels shape:", labels.shape)  

    print("Question 2 :")
    directory_path = r"C:\Users\LENOVO\OneDrive\Desktop\Master\Deep learning\TD1-Image-classification\image-classification\data\cifar-10-python\cifar-10-batches-py"
    data, labels = read_cifar(directory_path)
    print("Data shape:", data.shape)  
    print("Labels shape:", labels.shape)  

    print("Question 3 :")
    split = 0.8  # 80% training, 20% testing
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split)
    print("Training data shape:", data_train.shape)
    print("Training labels shape:", labels_train.shape)
    print("Testing data shape:", data_test.shape)
    print("Testing labels shape:", labels_test.shape)

    print("Implementation de Knn :")
    data, labels = read_cifar(directory_path) 
    plot_knn_accuracy_vs_k(data, labels, split_factor=0.9) 