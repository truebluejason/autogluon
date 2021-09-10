from matplotlib import scale
import matplotlib.pyplot as plt
from scipy.stats import t, gamma, norm, bernoulli
import numpy as np
import time


# marginal prior parameter over mu
def t_parameters(sample_mean, sample_var, n):
    mu, lamb, alpha, beta = sample_mean, n, n/2, n/(2/sample_var)
    df, sigma = 2 * alpha, np.sqrt(beta/(lamb*alpha))
    return df, mu, sigma


# sample_mean, sample_var, n should be floats
def sample_observations(sample_mean, sample_var, n, num_samples=1000):
    mu, lamb, alpha, beta = sample_mean, n, n/2, n/(2/sample_var)
    precision = gamma.rvs(alpha, scale=1/beta, random_state=0, size=num_samples)
    mean = norm.rvs(mu * np.ones(num_samples), 1/(lamb*precision))
    return norm.rvs(mean, 1/precision)


# sample_mean, sample_var, n should be floats
def posterior_t_parameters(sample_mean, sample_var, n, observations):
    mu, lamb, alpha, beta = sample_mean, n, n/2, n/(2/sample_var)
    obs_mean, obs_var, obs_size = observations.mean(), observations.var(), len(observations)
    posterior_mu = (lamb * mu + obs_size * obs_mean)/(lamb + obs_size)
    posterior_lamb = lamb + obs_size
    posterior_alpha = alpha + obs_size/2
    posterior_beta = beta + 0.5*(obs_size*obs_var + (lamb*obs_size*(obs_mean-mu)**2)/(lamb + obs_size))
    posterior_df, possterior_sigma = 2 * posterior_alpha, np.sqrt(posterior_beta/(posterior_lamb*posterior_alpha))
    return posterior_df, posterior_mu, possterior_sigma


# sample mean, sample_var, n should be floats
def expected_info_gain(sample_mean, sample_var, n, num_samples=1000):
    observations = sample_observations(sample_mean, sample_var, n, num_samples)
    prior_df, prior_mu, prior_sigma = t_parameters(sample_mean, sample_var, n)
    prior_probs = t.cdf(0, prior_df, loc=prior_mu, scale=prior_sigma)
    prior_entropy = bernoulli(prior_probs).entropy()
    posterior_entropies = []
    for obs in observations:
        posterior_df, posterior_mu, posterior_sigma = posterior_t_parameters(sample_mean, sample_var, n, np.array([obs]))
        posterior_probs = t.cdf(0, posterior_df, loc=posterior_mu, scale=posterior_sigma)
        posterior_entropy = bernoulli(posterior_probs).entropy()
        posterior_entropies.append(posterior_entropy)
    return prior_entropy - np.mean(posterior_entropy), prior_probs, posterior_probs

"""
# plot feature relevance probability of prior distribution
horizon = 25
true_mean = 0.00001 * np.ones(horizon)
true_var = 0.001 ** 2 * np.ones(horizon)
x = np.arange(1, horizon+1)
df, mu, sigma = t_parameters(true_mean, true_var, x)
y = t.cdf(0, df, loc=mu, scale=sigma)
print([round(p, 3) for p in y])
plt.plot(x, y)
plt.show()
"""

# print expected information gain for different parameter settings
horizon = 500

true_means = [100., 100.]#, 100., 100., 100.]
true_stds = [100., 10.]#, 1., 0.1, 0.01]
labels = ["0.1", "0.01"]#, "0.001", "0.0001", "0.00001"]
x = np.arange(1, horizon+1)
for true_mean, true_std, label in zip(true_means, true_stds, labels):
    true_var = true_std ** 2
    time_start = time.time()
    y = [expected_info_gain(true_mean, true_var, i, num_samples=1000)[0] for i in x]
    duration = time.time() - time_start
    print(f"({duration}s): {[round(el, 4) for el in y]}")
    plt.plot(x, y, label=label)
plt.title(f"EIG Curve (std: {true_std})")
plt.legend()
plt.show()

true_means = [0.1, 0.01, 0.001, 0.0001, 0.00001]
true_stds = [0.001, 0.001, 0.001, 0.001, 0.001]
labels = ["0.1", "0.01", "0.001", "0.0001", "0.00001"]
x = np.arange(1, horizon+1)
for true_mean, true_std, label in zip(true_means, true_stds, labels):
    true_var = true_std ** 2
    time_start = time.time()
    y = [expected_info_gain(true_mean, true_var, i, num_samples=10)[0] for i in x]
    duration = time.time() - time_start
    print(f"({duration}s): {[round(el, 4) for el in y]}")
    plt.plot(x, y, label=label)
plt.title(f"EIG Curve (std: {true_std})")
plt.legend()
plt.show()

true_means = [0.001, 0.001, 0.001, 0.001, 0.001]
true_stds =[0.1, 0.01, 0.001, 0.0001, 0.00001]
labels = ["0.1", "0.01", "0.001", "0.0001", "0.00001"]
x = np.arange(1, horizon+1)
for true_mean, true_std, label in zip(true_means, true_stds, labels):
    true_var = true_std ** 2
    time_start = time.time()
    y = [expected_info_gain(true_mean, true_var, i, num_samples=10)[0] for i in x]
    duration = time.time() - time_start
    print(f"({duration}s): {[round(el, 4) for el in y]}")
    plt.plot(x, y, label=label)
plt.title(f"EIG Curve (mean: {true_mean})")
plt.legend()
plt.show()


import pdb; pdb.set_trace()
expected_info_gain(-.001, .005**2, 3)
# expected_info_gain(0.01, 0.001**2, 10)
# [round(expected_info_gain(0.01, 0.001**2, i)[0], 3) for i in np.arange(1,26)]
# ig_one = expected_info_gain(true_mean, true_var, 10)
# ig_two = expected_info_gain(true_mean, true_var, 100)
# print(f"Info Gain 1: {round(ig_one,4)}, Info Gain 2: {round(ig_two,4)}")
