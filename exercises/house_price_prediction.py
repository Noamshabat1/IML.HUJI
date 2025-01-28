from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

averages = dict()
columns = []


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Remove redundant features
    X = X.drop(['id', 'date', 'lat', 'long'], axis=1)

    if y is not None:
        y = y.dropna()
        y = y[y > 0]
        X = X.loc[y.index]

        # Drops samples with invalid zipcodes (to avoid non-matching columns between train and test sets)
        X = X[X['zipcode'] > 0]
        y = y.loc[X.index]

        # Calculates averages
        averages['yr_built'] = np.mean([X['yr_built'] > 0])
        averages['sqft_living'] = np.mean([X['sqft_living'] > 0])
        averages['sqft_lot'] = np.mean([X['sqft_lot'] > 0])

        averages['sqft_living15'] = np.mean([X['sqft_living15'] >= 0])
        averages['sqft_lot15'] = np.mean([X['sqft_lot15'] >= 0])

        averages['bedrooms'] = np.mean([X['bedrooms'] >= 0])
        averages['bathrooms'] = np.mean([X['bathrooms'] >= 0])

        averages['floors'] = np.mean([X['floors'] >= 0])

    # Fix samples that are invalid (NaN/Negative fields/Values that don't match the field's description)
    X = X.fillna(0)

    X.loc[X['sqft_living'] <= 0, 'sqft_living'] = averages['sqft_living']
    X.loc[X['sqft_lot'] <= 0, 'sqft_lot'] = averages['sqft_lot']

    X.loc[X['sqft_living15'] < 0, 'sqft_living15'] = averages['sqft_living15']
    X.loc[X['sqft_lot15'] < 0, 'sqft_lot15'] = averages['sqft_lot15']

    X.loc[X['floors'] < 0, 'floors'] = averages['floors']

    X.loc[X['yr_built'] <= 0, 'yr_built'] = averages['yr_built']
    X.loc[X['yr_renovated'] < 0, 'yr_renovated'] = 0

    X.loc[X['sqft_basement'] < 0, 'sqft_basement'] = 0
    X.loc[X['sqft_above'] < 0, 'sqft_above'] = 0

    X.loc[X['bedrooms'] < 0, 'bedrooms'] = averages['bedrooms']
    X.loc[X['bedrooms'] > 10, 'bedrooms'] = 10
    X.loc[X['bathrooms'] < 0, 'bathrooms'] = averages['bathrooms']
    X.loc[X['bathrooms'] > 10, 'bathrooms'] = 10

    X.loc[X['waterfront'] < 0, 'waterfront'] = 0
    X.loc[X['waterfront'] > 1, 'waterfront'] = 1
    X.loc[X['condition'] < 1, 'condition'] = 1
    X.loc[X['condition'] > 5, 'condition'] = 5
    X.loc[X['view'] < 0, 'view'] = 0
    X.loc[X['view'] > 4, 'view'] = 4
    X.loc[X['grade'] < 1, 'grade'] = 1
    X.loc[X['grade'] > 15, 'grade'] = 15

    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    global columns

    if y is not None:
        columns = X.columns
    else:
        # Makes sure X has the same columns in both train and test
        X = X.reindex(columns=columns, fill_value=0)

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_std = np.std(Y)

    for feature in X.columns:
        if 'zipcode_' in feature: continue

        f = X[feature]

        # Calculates the Pearson Correlation of every feature and the result series
        # We care about the cov[0][1] (or cov[1][0]) because it holds the relation between the first & second vector
        pc = np.cov(f, y)[0][1] / (np.std(f) * y_std)

        # Plotting a scatter graph representing the relation between the feature and the response
        px.scatter(x=f, y=y, trendline="ols",
                   labels=dict(x=f"{feature}", y="Response"),
                   title=f"Pearson Correlation between {feature} and the response is {pc}") \
            .write_image(f"{output_path}/{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    X = df.drop(['price'], axis=1)
    Y = df['price']

    train_X, train_Y, test_X, test_Y = split_train_test(X, Y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_Y = preprocess_data(train_X, train_Y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_Y, './ex2_graphs')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_model = LinearRegression(True)

    # Processing the test data, and keeping only test samples that have a valid price, otherwise the squared error
    # won't be valid.
    test_Y = test_Y.dropna()
    test_X = test_X.loc[test_Y.index]
    test_X = preprocess_data(test_X, None)[0]

    # Calculating losses for every iteration of every percentage
    losses = np.empty(shape=(91, 10))

    for percentage in range(10, 101):
        for iteration in range(0, 10):
            fit_X = train_X.sample(frac=(percentage / 100))
            fit_Y = train_Y[fit_X.index]

            linear_model.fit(fit_X.to_numpy(), fit_Y)

            losses[(percentage - 10), iteration] = linear_model.loss(np.array(test_X), np.array(test_Y))

    # - mean(loss) = mean of every percentage value
    # - std(loss) = sqrt(mean(loss)) = variance(?)
    losses_mean = losses.mean(axis=1)
    losses_var = losses.std(axis=1)

    go.Figure([
        go.Scatter(x=np.arange(10, 101), y=losses_mean, name="Real Mean",
                   mode="markers+lines",
                   marker=dict(color="blue", opacity=1)),
        go.Scatter(x=np.arange(10, 101), y=(losses_mean - 2 * losses_var), name="Error Ribbon",
                   mode="markers+lines", fill='tonexty',
                   marker=dict(color="lightgrey", opacity=.5), line=dict(color="lightgrey")),
        go.Scatter(x=np.arange(10, 101), y=(losses_mean + 2 * losses_var), name="Error Ribbon",
                   mode="markers+lines", fill='tonexty',
                   marker=dict(color="lightgrey", opacity=.5), line=dict(color="lightgrey"))
    ],

        layout=go.Layout(
            title=f"Mean loss as a function of the percentage of data used for fitting the model",
            xaxis={"title": "Percentage of data used for fitting"},
            yaxis={"title": "Mean loss of the test set"})
    ).write_image("./ex2_graphs/percentage_graph.png")
