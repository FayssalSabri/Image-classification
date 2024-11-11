import numpy as np
import matplotlib.pyplot as plt
import os
from read_cifar import read_cifar, split_dataset
from knn import distance_matrix, knn_predict, evaluate_knn
from mlp import run_mlp_training, one_hot

def main():
    # Paramètres généraux
    directory_path = r"C:\Users\LENOVO\OneDrive\Desktop\Master\Deep learning\TD1-Image-classification\image-classification\data\cifar-10-python\cifar-10-batches-py"
    split_factor = 0.8  
    d_h = 64  # Nombre de neurones dans la couche cachée (MLP)
    learning_rate = 0.00001  # Taux d'apprentissage réduit pour éviter le surapprentissage
    num_epoch = 100
    patience = 10  # Patience pour l'arrêt anticipé

    print("Loading CIFAR-10 dataset...")
    data, labels = read_cifar(directory_path)

    print("Splitting the dataset...")
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split_factor)
    num_classes = 10
    labels_train = one_hot(labels_train, num_classes=num_classes)
    labels_test = one_hot(labels_test, num_classes=num_classes)

    print("Training the MLP model...")
    train_accuracies, test_accuracies = run_mlp_training(
        data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch, patience
    )

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/mlp.png")
    plt.show()

    print(f"Final test accuracy (MLP): {test_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()
