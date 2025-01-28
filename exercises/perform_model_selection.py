from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    # Selects 50 random elements for training and rest for testing
    training_indexes = np.random.choice(len(X), n_samples)
    testing_indexes = np.delete(np.arange(len(X)), training_indexes)

    train_X, train_y = X[training_indexes], y[training_indexes]
    test_X, test_y = X[testing_indexes], y[testing_indexes]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # lambdas = np.linspace(0.001, 2, n_evaluations)
    ridge_lambdas, lasso_lambdas = np.linspace(0.00001, 1, num=n_evaluations), np.linspace(.001, 1, num=n_evaluations)
    ridge_error, lasso_error = np.zeros(shape=(n_evaluations, 2)), np.zeros(shape=(n_evaluations, 2))

    for i, lam in enumerate(ridge_lambdas):
        ridge_error[i] = cross_validate(RidgeRegression(lam), train_X, train_y, scoring=mean_square_error, cv=5)

    for i, lam in enumerate(lasso_lambdas):
        lasso_error[i] = cross_validate(Lasso(lam, max_iter=5000), train_X, train_y, scoring=mean_square_error, cv=5)

    fig = make_subplots(1, 2, subplot_titles=["Ridge Regression", "Lasso Regression"], shared_xaxes=True)

    fig.add_traces([
        go.Scatter(x=ridge_lambdas, y=ridge_error[:, 0], name="Train Error (Ridge)"),
        go.Scatter(x=ridge_lambdas, y=ridge_error[:, 1], name="Validation Error (Ridge)"),
        go.Scatter(x=lasso_lambdas, y=lasso_error[:, 0], name="Train Error (Lasso)"),
        go.Scatter(x=lasso_lambdas, y=lasso_error[:, 1], name="Validation Error (Lasso)")
    ], rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])

    fig.update_layout(width=1000, height=500,
                      title=f"Mean Train and Validation Error over {5} "
                            f"folds, as a function of Regularization parameter")
    fig.update_xaxes(title="λ value (Regularization Parameter)")

    fig.write_image(f"ex4_graphs/lasso_ridge_cross_validate_errors.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_lambda = ridge_lambdas[np.argmin(ridge_error[:, 1])]
    lasso_lambda = lasso_lambdas[np.argmin(lasso_error[:, 1])]

    ridge = RidgeRegression(lam=ridge_lambda).fit(train_X, train_y)
    lasso = Lasso(alpha=lasso_lambda).fit(train_X, train_y)
    linear = LinearRegression().fit(train_X, train_y)

    print(f"Best λ (Regularization Parameter) for Ridge Regression: {np.round(ridge_lambda, 6)}")
    print(f"Best λ (Regularization Parameter) for Lasso Regression: {np.round(lasso_lambda, 6)}")

    print(f"Ridge Model (fitted with λ={np.round(ridge_lambda, 6)}) Error for the Test Set: "
          f"{np.round(ridge.loss(test_X, test_y), 2)}")
    print(f"Lasso Model (fitted with λ={np.round(lasso_lambda, 6)}) Error for the Test Set: "
          f"{np.round(mean_square_error(test_y, lasso.predict(test_X)), 2)}")
    print(f"Linear Model Error for the Test Set: "
          f"{np.round(linear.loss(test_X, test_y), 2)}")


if __name__ == '__main__':
    np.random.seed(0)

    select_regularization_parameter()

