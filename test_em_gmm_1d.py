#!/usr/bin/env python3

import em_gmm_1d
import numpy as np
import matplotlib.pyplot as plt


data = np.ones(1)*100

for i in range(2):
    data = np.concatenate((data, np.random.normal(i*10+10, 2*i+1, 100)))




mu, sigma = em_gmm_1d.em_gmm_1d(data, 10, False)

print(mu, sigma)

em_gmm_1d.plot_1d(data, mu, sigma, 'black')
