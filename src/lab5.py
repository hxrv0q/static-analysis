import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def compute_confidence_intervals(x, intercept, slope, s2, t):
    n = len(x)
    xmean = np.mean(x)

    delta = np.sqrt(s2) * np.sqrt(1 / n + xmean**2 / np.sum((x - xmean) ** 2))
    ci_intercept = [intercept - t * delta, intercept + t * delta]

    delta_slope = np.sqrt(s2) / np.sqrt(np.sum((x - xmean) ** 2))
    ci_slope = [slope - t * delta_slope, slope + t * delta_slope]

    return ci_intercept, ci_slope


def compute_variance_confidence_intervals(s2, n, alpha):
    chi2 = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], n - 2)
    return [(n - 2) * s2 / chi for chi in chi2[::-1]]


def plot_corridor_regression(x, yr, yleft, yright):
    plt.plot(x, yr, color="red", label="$y_r$")
    plt.plot(x, yleft, color="green", label="$yleft$", linestyle="dashed")
    plt.plot(x, yright, color="blue", label="$yright$", linestyle="dashed")
    plt.legend()
    plt.show()


def plot_regression(x, y, yr):
    plt.plot(x, yr, color="red", label="$y_r$")
    plt.scatter(x, y, label="$y_i$")
    plt.legend()
    plt.show()


def main():
    np.random.seed(0)

    # fmt: off
    x = np.array([ -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([ 2.258, 0.738, 1.479, 1.094, 1.177, 1.126, 0.523, 0.741, -0.364, 0.673, 0.259, -0.378, -0.568, -1.266, -1.376])
    # fmt: on

    slope, intercept = stats.linregress(x, y)[:2]
    yr = x * slope + intercept

    plot_regression(x, y, yr)

    n = len(x)
    alpha = 0.05
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    s2 = 1 / (n - 2) * np.sum((y - yr) ** 2)

    xmean = np.mean(x)
    se = np.sqrt(s2 * (1 / n + (x - xmean) ** 2 / np.sum((x - xmean) ** 2)))

    plot_corridor_regression(x, yr, yr + t * se, yr - t * se)

    f = stats.f.ppf(1 - alpha / 2, dfn=2, dfd=n - 2)

    plot_corridor_regression(x, yr, yr + 2 * f * se, yr - 2 * f * se)

    print(f"{slope=} {intercept=}")

    ci_intercept, ci_slope = compute_confidence_intervals(x, intercept, slope, s2, t)
    ci_var = compute_variance_confidence_intervals(s2, n, alpha)

    print(f"{ci_intercept=}\n{ci_slope=}\n{ci_var=}")


if __name__ == "__main__":
    main()
