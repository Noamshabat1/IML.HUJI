import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''

    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # In order to use train_y and test_y for colors and symbols, they must be of type int
    train_y = train_y.astype(int)
    test_y = test_y.astype(int)


    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    # Creating arrays holding the train & test errors per number of learners
    train_error = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    test_error = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]

    # Creating the graph displaying the 2 losses as function of the number of learners
    fig = go.Figure(
        layout=go.Layout(title="AdaBoost Loss as function of number of fitted learners", margin=dict(t=100)))

    fig.add_traces([
        go.Scatter(x=np.arange(n_learners), y=train_error, mode='lines', name="Train Error"),
        go.Scatter(x=np.arange(n_learners), y=test_error, mode='lines', name="Test Error")
    ])

    fig.update_layout(width=650, height=500) \
        .update_xaxes(title_text="Number of fitted learners used to evaluate") \
        .update_yaxes(title_text="Loss")

    fig.write_image(f"ex4_graphs/train_test_errors_noise_{noise}.png")

    # Question 2: Plotting decision surfaces

    # A helper function to prevent code dupes when creating a graph representing decision boundaries
    def create_decision_boundaries_traces_for_t(t):
        return [decision_surface(lambda xs: adaboost.partial_predict(xs, t), lims[0], lims[1], showscale=False),
                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=test_y, symbol=class_symbols[test_y],
                                       colorscale=[custom[0], custom[-1]], size=4,
                                       line=dict(color="black", width=1)))]

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[f"AdaBoost decision boundary for {t} iterations" for t in T],
                        horizontal_spacing=0.05, vertical_spacing=.03)

    for i, t in enumerate(T):
        fig.add_traces(create_decision_boundaries_traces_for_t(t), rows=1, cols=i + 1)

    fig.update_layout(width=2000, height=500)

    fig.write_image(f"ex4_graphs/decision_boundaries_noise_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    best_index = np.argmin(test_error)
    best_t = best_index + 1
    accuracy = 1 - test_error[best_index]

    fig = go.Figure(layout=go.Layout(title=f"Decision surface of the best performing ensemble<br>"
                                           f"size: {best_t}, accuracy: {accuracy}",
                                     margin=dict(t=100)))

    fig.add_traces(create_decision_boundaries_traces_for_t(best_t))

    fig.update_layout(width=500, height=500)

    fig.write_image(f"ex4_graphs/best_ensemble_size_noise_{noise}.png")

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(layout=go.Layout(title=f"Prediction on training data with {n_learners} iterations<br>"
                                           f"with size as function of weight in D_",
                                     margin=dict(t=100)))

    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=train_y, symbol=class_symbols[train_y],
                                   colorscale=[custom[0], custom[-1]],
                                   size=adaboost.D_/np.max(adaboost.D_) * 5 * 3,
                                   line=dict(color="black", width=1)))])

    fig.update_layout(width=500, height=500)

    fig.write_image(f"ex4_graphs/last_iteration_training_set_noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
