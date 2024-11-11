import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    # Forward pass
    a0 = data
    z1 = np.dot(a0, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    predictions = a2

    loss = np.mean(np.square(predictions - targets))
    
    dloss_da2 = 2 * (predictions - targets) / targets.shape[0]
    da2_dz2 = sigmoid_derivative(a2)
    dloss_dz2 = dloss_da2 * da2_dz2

    dloss_dw2 = np.dot(a1.T, dloss_dz2)
    dloss_db2 = np.sum(dloss_dz2, axis=0, keepdims=True)

    dloss_da1 = np.dot(dloss_dz2, w2.T)
    da1_dz1 = sigmoid_derivative(a1)
    dloss_dz1 = dloss_da1 * da1_dz1

    dloss_dw1 = np.dot(a0.T, dloss_dz1)
    dloss_db1 = np.sum(dloss_dz1, axis=0, keepdims=True)

    # Update weights and biases 
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2

    return w1, b1, w2, b2, loss


def one_hot(labels, num_classes=None):
    
    if num_classes is None:
        num_classes = np.max(labels) + 1
    
    one_hot_matrix = np.zeros((len(labels), num_classes), dtype=int)
    
    one_hot_matrix[np.arange(len(labels)), labels] = 1
    
    return one_hot_matrix


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    
   
    a0 = data
    z1 = np.dot(a0, w1) + b1
    z1 = np.clip(z1, -500, 500)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)  
    predictions = a2

    loss = -np.mean(np.sum(labels_train * np.log(predictions + 1e-15), axis=1))  # petit epsilon pour éviter log(0)

    dloss_dz2 = predictions - labels_train

    dloss_dw2 = np.dot(a1.T, dloss_dz2)
    dloss_db2 = np.sum(dloss_dz2, axis=0, keepdims=True)

    dloss_da1 = np.dot(dloss_dz2, w2.T)
    da1_dz1 = a1 * (1 - a1)  
    dloss_dz1 = dloss_da1 * da1_dz1

    dloss_dw1 = np.dot(a0.T, dloss_dz1)
    dloss_db1 = np.sum(dloss_dz1, axis=0, keepdims=True)

    # Mise à jour des poids
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2

    return w1, b1, w2, b2, loss


def compute_accuracy(predictions, labels):
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy * 100

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = []

    for epoch in range(num_epoch):
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        
        z1 = np.dot(data_train, w1) + b1
        z1 = np.clip(z1, -500, 500)
        a1 = 1 / (1 + np.exp(-z1))  
        z2 = np.dot(a1, w2) + b2
        predictions = softmax(z2)  

        
        accuracy = compute_accuracy(predictions, labels_train)
        train_accuracies.append(accuracy)

        
        print(f"Epoch {epoch + 1}/{num_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):

    z1 = np.dot(data_test, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))  
    z2 = np.dot(a1, w2) + b2
    predictions = softmax(z2)   

    test_accuracy = compute_accuracy(predictions, labels_test)

    return test_accuracy


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate=0.001, num_epoch=100, patience=10):
    d_in = data_train.shape[1]
    d_out = labels_train.shape[1]
    
    w1 = np.random.randn(d_in, d_h) * 0.01
    b1 = np.zeros((1, d_h))
    w2 = np.random.randn(d_h, d_out) * 0.01
    b2 = np.zeros((1, d_out))

    train_accuracies = []
    test_accuracies = []

    best_test_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        
        z1_train = np.clip(np.matmul(data_train, w1) + b1, -500, 500)
        a1_train = np.maximum(0, z1_train)  
        z2_train = np.clip(np.matmul(a1_train, w2) + b2, -500, 500)
        a2_train = np.exp(z2_train - np.max(z2_train, axis=1, keepdims=True))
        a2_train /= np.sum(a2_train, axis=1, keepdims=True)  # Softmax activation

        train_predictions = np.argmax(a2_train, axis=1) == np.argmax(labels_train, axis=1)
        train_accuracy = np.mean(train_predictions) * 100
        train_accuracies.append(train_accuracy)

        test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epoch} - Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_accuracies, test_accuracies
