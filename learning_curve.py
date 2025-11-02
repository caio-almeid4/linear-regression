import matplotlib.pyplot as plt
import numpy as np

from linear_regression import LinearRegression


def plot_learning_curve(lr: LinearRegression, X: np.ndarray, y: np.ndarray) -> None:
    
    sizes = np.linspace(0.01, 1, 40)
    rmse_train = []
    rmse_test = []
    train_sizes = []
    
    for train_size in sizes:
        
        size = int(len(X) * train_size)
        train_sizes.append(size)
        X_train, y_train = X[:size], y[:size]
        if train_size != 1:
            X_test, y_test = X[size:], y[size:]
        else:
            X_test, y_test = X_train, y_train
        lr.fit(X_train, y_train)
        rmse_train.append(lr.rmse)
        
        y_predict = lr.predict(X_test)
        rmse_test.append(lr.get_rmse(y_test, y_predict))
        
    
    plt.plot(train_sizes, rmse_train, 'r-+', linewidth=2, label='train')
    plt.plot(train_sizes, rmse_test, 'b-', linewidth=3, label='test')
    plt.xlabel('train_size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.plot()
        
    