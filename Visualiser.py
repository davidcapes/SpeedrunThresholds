import numpy as np
from scipy.stats import expon, gamma, norm, uniform
import matplotlib.pyplot as plt

pdfs = (
    lambda x: norm.pdf(x, loc=15, scale=np.sqrt(12)),
    lambda x: uniform.pdf(x, loc=13, scale=18),
    lambda x: expon.pdf(x, scale=20),
    lambda x: (6 * (x - 13) ** 2) / (np.pi * ((x - 13) ** 6 + 4)),
    lambda x: (norm.pdf(x, loc=13, scale=5) + norm.pdf(x, loc=14, scale=2) + norm.pdf(x, loc=17, scale=0.5)) / 3,
    lambda x: gamma.pdf(x, a=5, scale=3),
)

cdfs = (
    lambda x1: norm.cdf(x1, loc=15, scale=np.sqrt(12)),
    lambda x2: uniform.cdf(x2, loc=13, scale=18),
    lambda x3: expon.cdf(x3, scale=20),
    lambda x4: np.arctan((x4 - 13) ** 3 / 2) / np.pi + 1 / 2,
    lambda x5: (norm.cdf(x5, loc=13, scale=5) + norm.cdf(x5, loc=14, scale=2) + norm.cdf(x5, loc=17, scale=0.5)) / 3,
    lambda x6: gamma.cdf(x6, a=5, scale=3)
)

colors = ["blue", "green", "red", "orange", "darkturquoise", "purple"]

statistics = ["mean=15\nvariance=12\nmedian=15\nIQR=4.67",
              "mean=22\nvariance=27\nmedian=22\nIQR=9",
              "mean=20\nvariance=400\nmedian=13.86\nIQR=21.97",
              "mean=13\nvariance=3.17\nmedian=13\nIQR=2.52",
              "mean=14.67\nvariance=12.64\nmedian=15.65\nIQR=4.26",
              "mean=15\nvariance=45\nmedian=14\nIQR=8.72"]
locations = ["upper right", "upper left", "upper right", "upper right", "upper right", "upper right"]

for i, p in enumerate(pdfs):
    x = np.linspace(0, 35, 3000)
    y = np.array([p(i) for i in x])
    plt.plot(x, y, '-', color=colors[i])
    plt.fill_between(x, y, color=colors[i], alpha=0.2)
    plt.grid(visible=True, linestyle=":", linewidth=0.5, color="gray")
    plt.title(f"Visualization of the pdf for Task {i + 1}.", fontsize=16)
    plt.legend([statistics[i]], loc=locations[i])
    plt.savefig(f"Figures/Task{i + 1}.png")
    plt.cla()
