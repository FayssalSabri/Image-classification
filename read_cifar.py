import numpy as np
import pickle
import os

def read_cifar_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        data = np.array(data, dtype=np.float32) / 255.0  # Normalisation
        labels = np.array(labels, dtype=np.int64)
    return data, labels

def read_cifar(directory_path):
    data_list = []
    labels_list = []
    
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]

    for batch_file in batch_files:
        batch_path = os.path.join(directory_path, batch_file)
        with open(batch_path, 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
            data = np.array(data, dtype=np.float32) / 255.0  # Normalisation
            labels = np.array(labels, dtype=np.int64)
            data_list.append(data)
            labels_list.append(labels)

    data = np.vstack(data_list)
    labels = np.hstack(labels_list)
    
    return data, labels

def split_dataset(data, labels, split):
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_index = int(num_samples * split)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    data_train = data[train_indices]
    labels_train = labels[train_indices]
    data_test = data[test_indices]
    labels_test = labels[test_indices]
    return data_train, labels_train, data_test, labels_test
