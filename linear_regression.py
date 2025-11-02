import numpy as np


class LinearRegression:
    """Class that implements a simple Linear Regression,
    using batch gradient descent
    """    
    
    
    def __init__(self):
        self.gd_epochs = 1000
        self.rmse = 0

    def fit(
        self, X: np.ndarray, 
        y: np.ndarray, 
        learning_rate: float = 0.1
        ) -> None:
        """Train the model with given data

        Args:
            X (np.array): Matrix with training feature data.
            y (_type_): Array with training labels.
            learning_rate (float, optional): Learning rate to apply.
            batch gradient descent. Defaults to 0.1.
        """        
        X_with_dummy = self._get_matrix_with_dummy(X)
        theta = self._batch_gradient_descent(X_with_dummy, y, learning_rate)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
        self.rmse = self.get_rmse(y, self.predict(X))
        
        return None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the estimated parameters to predict the output
        for a given input.

        Args:
            X (np.array): Matrix with training feature data.

        Returns:
            prediction (np.ndarray): Array with predicted values.
        """        
        
        X_with_dummy = self._get_matrix_with_dummy(X)
        theta = np.concat(([self.intercept_], self.coef_), axis=0)
        prediction = X_with_dummy @ theta
        
        return prediction
        
    def _batch_gradient_descent(
        self, X: np.ndarray, 
        y: np.ndarray, 
        learning_rate: float
        ) -> np.ndarray:
        """This function estimates theta parameters using 
        batch gradient descent over Mean Squared Error cost
        function.

        Args:
            X (np.ndarray): Matrix with training feature data.
            y (np.ndarray): Array with training labels.
            learning_rate (float): Learning rate to apply.

        Returns:
            theta (np.ndarray): Array with estimated parameters.
        """        
        
        m = len(X)
        n = len(X[0])
        theta = np.random.randn(n, 1)
        
        for _ in range(self.gd_epochs):
            gradient = 2 / m * X.T @ (X @ theta - y)
            theta = theta - learning_rate * gradient
            
        return theta
    
    def _get_matrix_with_dummy(self, X: np.ndarray) -> np.ndarray:
        """This function adds 1 at the beginning of every line from matrix X

        Args:
            X (nd.ndarray): Matrix with training feature data.

        Returns:
            X_with_dummy (np.ndarray): Output explained on description
        """            
        array_with_ones = np.full((len(X), 1), 1)
        X_with_dummy = np.concat((array_with_ones, X), axis=1)
        
        return X_with_dummy
    
    def get_rmse(self, y, y_predict) -> float:
        
        m = len(y)
        rmse = 1 / m * sum([(y_predict[i] - y[i])**2 for i in range(m)])
        
        return rmse
        
        
