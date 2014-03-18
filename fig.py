import matplotlib.pyplot as plt
import math
import numpy as np

plot_deg = np.linspace(.9, 10.1, 93)
plot_prob = np.power(plot_deg, -2 * np.ones(plot_deg.shape))
plot_prob = plot_prob / np.sum(plot_prob)

deg = range(1,11)
prob = [p for p in plot_prob[1::10]]

plt.plot(plot_deg, plot_prob, 'r', linewidth=3)
plt.scatter(deg, prob, 55, 'b')
for i in range(0, 10):
    plt.plot([deg[i], deg[i]], [0, prob[i]], 'b--', linewidth=3)
plt.xlabel('Degre')
plt.ylabel('Probabilite')
plt.xlim((.7, 10.1))
plt.ylim((0,1.1*plot_prob[0]))
plt.show()
