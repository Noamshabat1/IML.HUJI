from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)

    # Question 1 - Draw samples and print fitted model
    uni_gaus = UnivariateGaussian()
    uni_gaus.fit(samples)

    print(f'({uni_gaus.mu_}, {uni_gaus.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    estimations_error = [np.abs(uni_gaus.fit(samples[:sample_size]).mu_ - mu) for sample_size in range(10, 1001, 10)]

    figure1 = px.line(x=range(10, 1001, 10), y=np.array(estimations_error),
                      labels=dict(x="Sample Size", y="Estimation Error (Distance from real MU)"),
                      title="Estimation Error as a function of Sample Size")
    figure1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = uni_gaus.pdf(samples)

    figure2 = px.scatter(x=samples, y=pdf_values,
                         labels=dict(x="Sample Value", y="Calculated PDF Value"),
                         title="PDF Values of a set of 1000 samples distributed Normal(10, 1)")
    figure2.show()


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)

    # Question 4 - Draw samples and print fitted model
    mult_gaus = MultivariateGaussian()
    mult_gaus.fit(samples)

    print(f'Estimated Expectation: {mult_gaus.mu_}')
    print(f'Estimated Covariance Matrix: {mult_gaus.cov_}')

    # Question 5 - Likelihood evaluation
    linspace = np.linspace(-10, 10, 200)

    log_likelihood_values = np.array([
        [MultivariateGaussian.log_likelihood(
            np.array([linspace[i], 0, linspace[j], 0]), cov, samples) for i in range(200)]
        for j in range(200)])

    figure1 = px.imshow(log_likelihood_values.T, x=linspace, y=linspace,
                        labels={"x": "f3", "y": "f1"},
                        title="Normal Multivariate Distribution as a function of F3/F1, displayed in a Heat-Map")

    # Reverses the y-axis
    figure1.update_yaxes(autorange=True)
    figure1.show()

    # Question 6 - Maximum likelihood
    max_value = np.max(log_likelihood_values)
    max_indexes = np.where(log_likelihood_values == max_value)
    print(f'The maximum value is attained at (f3: {np.round(linspace[max_indexes[1]][0], 3)}, \
f1: {np.round(linspace[max_indexes[0]][0], 3)}), and gets the value {np.round(max_value, 3)}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
