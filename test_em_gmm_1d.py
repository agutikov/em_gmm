#!/usr/bin/env python3

import em_gmm_1d
import numpy as np
import matplotlib.pyplot as plt


#data = np.ones(1)*400
data = np.empty(1)

for i in range(3):
    data = np.concatenate((data, np.random.normal(i*10+100, 20*i+1, 4)))




mu, sigma = em_gmm_1d.em_gmm_1d(data, 1, True)

print(mu, sigma)

em_gmm_1d.plot_1d(data, mu, sigma, 'y')

m = np.median(data)
s = np.std(data)
print(m, s)
em_gmm_1d.plot_1d(data, [m], [s], 'b')
