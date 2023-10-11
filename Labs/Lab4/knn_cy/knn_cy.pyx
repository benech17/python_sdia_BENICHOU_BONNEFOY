# knn.pyx

import numpy as np

def knn_classification(double[:, ::1] X_train, int[:] y_train, double[:, ::1] X_test, int k):
    cdef int n_train = X_train.shape[0]
    cdef int n_test = X_test.shape[0]

    cdef int[:] class_pred = np.zeros(n_test, dtype=np.int32)


    cdef double[:,::1] X_test_view = X_test
    cdef double[:,::1] X_train_view = X_train
    cdef int[:] y_train_view = y_train
    
    cdef int i, j

    cdef double[:] distances = np.zeros(n_train, dtype=np.float64)
    cdef long long[:] nearest_indices = np.zeros(k, dtype=np.int64)
    cdef int[:] nearest_labels= np.zeros(k, dtype=np.int32)
    
    cdef int[:] y_pred= np.zeros(n_test, dtype=np.int32)

    for i in range(n_test):
        for j in range(n_train):
            distances[j] = 0.0
            for d in range(X_train.shape[1]):
                distances[j] += (X_test[i, d] - X_train[j, d]) * (X_test[i, d] - X_train[j, d])

        nearest_indices = np.argsort(distances)[:k]

        for j in range(k):
            
            nearest_labels[j] = y_train_view[nearest_indices[j]]
        
        y_pred[i] = np.argmax(np.bincount(nearest_labels))

    return y_pred
