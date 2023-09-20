import numpy as np
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_cent = X - X.mean(axis=0)
        self.U, self.S, self.V = np.linalg.svd(X_cent, full_matrices=False)

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        Xbar = data - data.mean(axis=0)
        Vk = self.V[:K]
        
        return Xbar @ Vk.T

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        K = 0
        target = np.sum(self.S) * retained_variance
        cumsum = np.cumsum(self.S)
        for i in range(len(cumsum)):
            if cumsum[i] > target:
                K = i
                break
        
        return self.transform(data,K)

    def get_V(self) -> np.ndarray:
        """ Getter function for value of V """

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig=None) -> None:  # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """
        self.fit(X)
        X_new = self.transform(X,2)
        zerox, onex, twox = [], [], []
        zeroy, oney, twoy = [], [], []
        
        for i in range(y.shape[0]):
            if y[i] == 0:
                zerox.append(X_new[i][0])
                zeroy.append(X_new[i][1])
            elif y[i] == 1:
                onex.append(X_new[i][0])
                oney.append(X_new[i][1])
            elif y[i] == 2:
                twox.append(X_new[i][0])
                twoy.append(X_new[i][1])
         
        plt.scatter(zerox, zeroy, c ="blue",
            linewidths = 2,
            marker ="s",
            edgecolor ="blue",
            s = 50)
 
        plt.scatter(onex, oney, c ="magenta",
            linewidths = 2,
            marker ="^",
            edgecolor ="magenta",
            s = 50)
    
        plt.scatter(twox, twoy, c ="red",
            linewidths = 2,
            marker ="^",
            edgecolor ="red",
            s = 50)
        

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend(['zero', 'one', 'two'])
        plt.show()
