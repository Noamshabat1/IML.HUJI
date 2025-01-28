import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=['Date'])
    df.fillna(0)

    # We can drop data that is too odd to be real (samples with lower temperature than -40 degrees)
    df = df[df['Temp'] > -40]

    # Adding the DayOfYear column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # Extracting only Israel's data (and converting Year to be string for discrete color scaling)
    israel_data = df[df['Country'] == 'Israel']
    israel_data = israel_data.astype({'Year': 'str'})

    # First graph/figure
    fig = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                     title="Israel's Temperature as a function of the day of the year")
    fig.update_xaxes(title_text="Day of the Year")
    fig.update_yaxes(title_text="Temperature")
    fig.write_image(f"./ex2_graphs/israel_temp_scatter.png")

    # Second graph/figure (grouping by Month, then applying std() on the Temperatures of each month)
    fig = px.bar(x=np.arange(1, 13), y=israel_data.groupby(["Month"])['Temp'].std(),
                 title="Israel's Temperature Standard Deviation by Month")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Standard Deviation")
    fig.write_image(f"./ex2_graphs/israel_temp_bar.png")

    # Question 3 - Exploring differences between countries
    mean_country_month_temp = df.groupby(["Country", "Month"])['Temp'].mean()
    std_country_month_temp = df.groupby(["Country", "Month"])['Temp'].std()

    fig = px.line(df.groupby(["Country", "Month"], as_index=False) \
                  .agg(avg=("Temp", "mean"), std=("Temp", "std")),
                  x="Month", y="avg", error_y="std", color="Country",
                  title="Average & Standard Deviation of Temperatures per Month, by Country")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Average Temperature")
    fig.write_image(f"./ex2_graphs/israel_temp_line.png")

    # Question 4 - Fitting model for different values of k
    israel_train_X, israel_train_Y, israel_test_X, israel_test_Y = \
        split_train_test(israel_data["DayOfYear"], israel_data["Temp"])

    losses = np.empty(shape=(10,))

    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(israel_train_X.to_numpy(), israel_train_Y.to_numpy())

        losses[k - 1] = np.round(poly_model.loss(israel_test_X.to_numpy(), israel_test_Y.to_numpy()), 2)
        print(f"Loss for polynomial degree of {k} is: {losses[k - 1]}")

    fig = px.bar(x=range(1, 11), y=losses, text=losses,
                 title="Loss value per polynomial degree")
    fig.update_xaxes(title_text="Polynom Degree")
    fig.update_yaxes(title_text="Loss")
    fig.write_image(f"./ex2_graphs/israel_temp_poly_bar.png")

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(israel_data["DayOfYear"].to_numpy(), israel_data["Temp"].to_numpy())

    country_error_df = pd.DataFrame([{"country": country,
                                      "loss": np.round(poly_model.loss(df[df["Country"] == country]["DayOfYear"],
                                                                       df[df["Country"] == country]["Temp"]), 2)}
                                     for country in ["Jordan", "South Africa", "The Netherlands"]])

    fig = px.bar(country_error_df, x="country", y="loss", color="country", text="loss",
                 title="Loss values per country for a model of degree 5, fitted with Israel data")
    fig.update_xaxes(title_text="Country")
    fig.update_yaxes(title_text="Loss")
    fig.write_image(f"./ex2_graphs/israel_temp_country_bar.png")
