import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error


import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc
from utils import custom

c = [custom[0], custom[-1]]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    recorded_values = []
    recorded_weights = []

    def gd_callback(solver, weights, val, grad, t, eta, delta):
        recorded_values.append(val)
        recorded_weights.append(weights)

    return gd_callback, recorded_values, recorded_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):

    empty = np.empty(shape=(0,))

    for eta in etas:
        for module, name in [(L1, "L1"), (L2, "L2")]:
            gd_callback, values, weights = get_gd_state_recorder_callback()

            gd = GradientDescent(learning_rate=FixedLR(eta), callback=gd_callback)
            gd.fit(module(weights=init), X=empty, y=empty)

            # Question 1: Plot the descent path
            fig = plot_descent_path(module=module,
                                    descent_path=np.array(weights),
                                    title=f"of an {name} module with fixed learning rate η={eta}")
            fig.write_image(f"./ex5_graphs/{name}_{eta}_descent.png")

            # Question 3: Plot the convergence rate
            fig = go.Figure([go.Scatter(y=values, mode="markers")],
                            layout=go.Layout(title=f"Convergence Rate of an {name} module with " 
                                                   f"fixed learning rate η={eta}"))

            fig.update_layout(width=650, height=500) \
                .update_xaxes(title_text="Iteration") \
                .update_yaxes(title_text=f"Convergence (w {name} norm)")

            fig.write_image(f"./ex5_graphs/{name}_{eta}_convergence.png")

            # Question 4: Lowest loss achieved
            print(f"The lowest loss achieved by an {name} module with a fixed learning rate η={eta} "
                  f"is {np.round(np.min(values), 3)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    empty = np.empty(shape=(0,))

    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for gamma in gammas:
        gd_callback, values, weights = get_gd_state_recorder_callback()

        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=gd_callback)
        gd.fit(L1(weights=init), X=empty, y=empty)

        # Plot algorithm's convergence for the different values of gamma
        pass

        # Plot descent path for gamma=0.95
        pass


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """

    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression(include_intercept=True).fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1],
                         mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),

              go.Scatter(x=fpr, y=tpr,
                         mode='markers+lines',
                         name="FPR / TPR",
                         marker_color=c[1][1])],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve and AUC of a Logistic Regression Model}}="
                  rf"{auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    fig.write_image("./ex5_graphs/logistic_regression_roc.png")

    test_loss = LogisticRegression(include_intercept=True, alpha=thresholds[np.argmax(tpr - fpr)])\
            .fit(X_train, y_train)\
            .loss(X_test, y_test)

    print("Best alpha for (True Positive - False Position) ratio is:", np.round(thresholds[np.argmax(tpr - fpr)], 3),
          f"with a test loss of {np.round(test_loss, 3)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    gd = GradientDescent(max_iter=20000, learning_rate=FixedLR(base_lr=1e-4))

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for penalty in ["l1", "l2"]:
        validation_scores = []

        for lam in lambdas:
            lr_model = LogisticRegression(solver=gd, penalty=penalty, lam=lam)

            train_score, validation_score = cross_validate(lr_model, X_train, y_train, scoring=misclassification_error)

            validation_scores.append(validation_score)

        best_lambda = lambdas[np.argmin(validation_scores)]

        lr_model = LogisticRegression(solver=gd, penalty=penalty, lam=best_lambda)

        print(f"When using a {penalty} penalty, the best lambda found is λ={best_lambda} "
              f"with test error of {np.round(lr_model.fit(X_train, y_train).loss(X_test, y_test), 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates() # Optional
    fit_logistic_regression()
