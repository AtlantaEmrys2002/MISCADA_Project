import numpy as np

import matplotlib.pyplot as plt

poly_p = np.polynomial.Polynomial([-26.32681683834449, -1.2813737892102322, -1.888182538356475, -2.147009478562273])

plt.plot(np.linspace(5, 9, 100), poly_p(np.linspace(5, 9, 100)))

plt.savefig("testing.png")

plt.close()