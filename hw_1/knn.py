import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        result_matrix_dist = np.zeros((X.shape[0], self.train_X.shape[0]))
        for matrix in range(X.shape[0]):
            for second_matrix in range(self.train_X.shape[0]):
                result_matrix_dist[matrix, second_matrix] = np.linalg.norm(self.train_X[second_matrix] - X[matrix], ord = 1)
                
        return result_matrix_dist


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        result_matrix_dist = np.zeros((X.shape[0], self.train_X.shape[0]))
        for matrix in range(X.shape[0]):
            result_matrix_dist[matrix] = np.linalg.norm(self.train_X - X[matrix], ord = 1, axis = 1)
            
        return result_matrix_dist
        


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        return np.linalg.norm(X.reshape(X.shape[0], 1, X.shape[1]) - self.train_X, ord=1, axis=2)


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        sort_dist = np.argsort(distances, axis = 1)[:, :self.k]
        class_search = np.take(self.train_y, sort_dist).astype(int)
        vals = np.apply_along_axis(np.bincount, 1, class_search)
        return np.apply_along_axis(np.argmax, 1, vals)

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        sort_dist = np.argsort(distances, axis = 1)[:, :self.k]
        class_search = np.take(self.train_y, sort_dist).astype(int)
        vals = [np.bincount(class_search[i]) for i in range(class_search.shape[0])]
        predict = list(map(np.argmax, vals))
        return np.array(predict)