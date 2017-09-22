
import numpy as np
import math
import matplotlib.pyplot as plt
from pprint import pprint


def plot_1d(X, mu, sigma, color):
    fig, ax1 = plt.subplots()
    ax1.hist(X, 100, facecolor='g', alpha=0.75)
    ax2 = ax1.twinx()
    N = 1000
    outer = np.zeros(N)
    for k in range(len(mu)):
        x = np.linspace(min(X)-10, max(X)+10, N)
        y = _normpdf(x, mu[k], sigma[k])
        ax2.plot(x, y, color, linewidth=1)
        outer = outer + y
    ax2.plot(x, outer, color, linewidth=3)
    plt.show()


def _normpdf(X, m, v):
    return np.array([math.exp(-((x-m)**2/(2*v**2)))/(v*math.sqrt(2*math.pi)) for x in X])

# EM (Expectation Maximazation) for 1-Dimentional data
def em_gmm_1d(X, K, plot_steps=False):
    X = sorted(X)
    N = len(X)
    rng = max(X)-min(X)

    # Use 1-d MLE for estimating initial values of mu and sigma
    Mu = np.mean(X)
    Sigma = np.std(X)
    
    # initial values
    mu = np.linspace(Mu-Sigma*3, Mu+Sigma*3, K)
    sigma = np.ones(K)*Sigma
    z = np.ones((K, N))

    # difference between steps
    d_mu = Sigma*6
    d_sigma = rng

    # loop condition tresholds
    d_mu_tres = rng/N
    d_sigma_tres = Sigma/math.sqrt(N)

    if plot_steps:
        plt.hist(X, 100, facecolor='g', alpha=0.75)
        plt.show()

    while d_mu > d_mu_tres and d_sigma > d_sigma_tres:
        # plot initial gaussians
        if plot_steps:
            plot_1d(X, mu, sigma, 'black')

        # E-step
        for k in range(K):
            dd = np.zeros(N)
            for i in range(K):
                dd = dd + _normpdf(X, mu[i], sigma[i])
            z[k] = _normpdf(X, mu[k], sigma[k]) / dd

        old_mu = mu.copy()
        old_sigma = sigma.copy()

        # M-step
        for k in range(K):
            Zk = sum(z[k])
            mu[k] = (1/Zk) * sum(z[k]*X)
            sigma[k] = np.sqrt((1/Zk) * sum(z[k] * (X-mu[k])**2))
        
        #if plot_steps:
        #    plot_1d(X, mu, sigma, 'red')

        d_mu = max(abs(mu - old_mu))
        d_sigma = max(abs(sigma - old_sigma))


    return mu, sigma












