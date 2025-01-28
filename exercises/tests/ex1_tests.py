import numpy as np
import plotly.express as px

import sys
sys.path.append("../../../../")

from IMLearn.learners import gaussian_estimators as ge
from utils import *
from scipy.stats import multivariate_normal
import math


print("uni.fit")
max_diff = 3
mu = 5; std = 10; m = 100000
samples = np.random.normal(loc=mu, scale=std, size=m)
fitted = ge.UnivariateGaussian(biased_var=False).fit(samples)
fitted_biased = ge.UnivariateGaussian(biased_var=True).fit(samples)
print(f"expected mu: {mu}")
print(f"our mu:      {fitted.mu_}\n")
print(f"expected var: {std ** 2}")
print(f"our var:      {fitted.var_}\n")

assert isinstance(fitted.mu_, float)
assert isinstance(fitted.var_, float)
assert isinstance(fitted_biased.mu_, float)
assert isinstance(fitted_biased.var_, float)

assert np.all(np.abs(fitted.mu_ - mu) < max_diff)
assert np.allclose(fitted_biased.mu_, fitted.mu_)
assert np.all(np.abs(fitted_biased.var_ - std ** 2) < max_diff)
assert(fitted_biased.var_ / (m - 1) == fitted.var_ / m)


print("uni.pdf")
max_diff = 0.001
mu = 5; std = 2
fitted = ge.UnivariateGaussian(biased_var=False).fit(np.random.normal(loc=mu, scale=std, size=10000000))
x = mu + np.random.rand(1000)
expected = multivariate_normal.pdf(x, mean=mu, cov=std ** 2);
our = fitted.pdf(x)
print(f"expected: {expected[:5]}")
print(f"our:      {our[:5]}\n")
assert np.all(np.abs(expected - our) < max_diff)
assert our.shape == x.shape

print("uni.log_likelihood")
x = mu + np.random.rand(10)
expected = np.log(np.prod(multivariate_normal.pdf(x, mean=mu, cov=std ** 2)))
our = fitted.log_likelihood(mu, std ** 2, x)
print(f"expected: {expected}")
print(f"our:      {our}\n")
assert np.allclose(expected, our)
assert isinstance(our, float)


print("multi.fit")
max_diff = 0.3
mu = np.array([5, -15]);
cov = np.array([
    [10, 3],
    [3,  40],
])
m = 1000000

samples = np.random.multivariate_normal(mu, cov, size=m)
fitted = ge.MultivariateGaussian().fit(samples)
print(f"expected mu:  {mu}")
print(f"our mu:       {fitted.mu_}\n")
print(f"expected var: \n {cov}")
print(f"our var:      \n {fitted.cov_}\n") # TODO: should be biased or not

assert np.all(np.abs(fitted.mu_ - mu) < max_diff)
assert np.all(np.abs(fitted.cov_ - cov) < max_diff)
assert fitted.mu_.shape == mu.shape
assert fitted.cov_.shape == cov.shape


print("multi.pdf")
x = mu + np.random.rand(10, 2)
expected = multivariate_normal.pdf(x, mean=mu, cov=cov);
our = fitted.pdf(x)
print(f"expected: {expected[:5]}")
print(f"our:      {our[:5]}\n")
assert np.all(np.abs(expected - our) < max_diff)
assert our.shape == (len(x),)


print("multi.log_likelihood")
print((multivariate_normal.pdf(x, mean=mu, cov=cov)))
expected = np.log(np.prod(multivariate_normal.pdf(x, mean=mu, cov=cov)))
our = fitted.log_likelihood(mu, cov, x)
print(f"expected: {expected}")
print(f"our:      {our}")
assert np.allclose(expected, our)
assert isinstance(our, float)

print("\nsuccess.")
