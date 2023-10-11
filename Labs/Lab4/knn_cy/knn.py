import bottleneck
import numpy as np

def knn_classification(x_train, class_train, x_test, n_neighbours=3):
    """
    K-nearest neighbours algorithm for classification.

    Parameters:
    - x_train: Training data
    - class_train: Class labels for training data
    - x_test: Test data
    - n_neighbours: Number of neighbours to consider

    Returns:
    - class_pred: Predicted class labels for test data
    """
    
    class_pred = np.empty(x_test.shape[0], dtype=class_train.dtype)
    
    for row, x in enumerate(x_test):
        distances = np.linalg.norm(x_train - x[np.newaxis, :], axis=1)  
        index = bottleneck.argpartition(distances, n_neighbours)[:n_neighbours]
        class_pred[row] = np.bincount(class_train[index]).argmax()

    return class_pred