import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def compute_statistics(data: np.ndarray) -> dict:
    return {
        "min": np.min(data),
        "max": np.max(data),
        "range": np.ptp(data),
        "n": len(data),
        "mean": np.mean(data),
        "var": np.var(data),
        "std": np.std(data),
        "skew": stats.skew(data),
        "kurtosis": stats.kurtosis(data),
    }


def plot_histogram(data: np.ndarray, bins: int):
    plt.hist(data, bins, alpha=0.7, edgecolor="black")
    plt.title(f"Гістограма m={bins}")
    plt.show()


def plot_polygon(data: np.ndarray, bins: int, cumulative: bool = False, normalized: bool = False):
    frequency, bin_edges = np.histogram(data, bins)
    frequency = frequency / len(data) if normalized else frequency
    frequency = np.cumsum(frequency) if cumulative else frequency

    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.plot(bin_mids, frequency, marker="o", linestyle="-")
    plt.xlabel("x")
    plt.show()


def plot_corridor(data: np.ndarray, alpha: float = 0.05):
    sorted_data, n = np.sort(data), len(data)
    z_alpha, ci = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)

    plt.plot(sorted_data, np.arange(1, n + 1) / n - ci, linestyle="--", color="red")
    plt.plot(sorted_data, np.arange(1, n + 1) / n + ci, linestyle="--", color="green")
    plt.xlabel("x")
    plt.show()


def find_N(z, eps=1e-3):
    return math.floor((1.0 / z) * np.sqrt(0.5 * np.log(1.0 / eps))) + 1


def K(z, eps=1e-3):
    if z <= 0:
        return 0

    n = find_N(z, eps)

    def f(k):
        return (-1) ** k * np.exp(-2 * k**2 * z**2)

    return np.sum([f(k) for k in range(-n, n + 1)])


def plot(x, y, xlabel, ylabel, title, axhline=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=ylabel)
    if axhline is not None:
        plt.axhline(axhline, color="red", linestyle="--", label="$1 - \\alpha$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_kolmogorov():
    z = np.linspace(0.1, 2, 500)
    eps = 1e-3
    k = [K(z1, eps) for z1 in z]
    alpha = 0.05
    plot(z, k, "z", "$K(z)$", "Критерій Колмогорова", axhline=1 - alpha)
    n = [find_N(z1, eps) for z1 in z]
    plot(z, n, "z", "$N(z)$", f"$N(z)$ при $\\varepsilon=${eps}")


def main():
    np.random.seed(0)
    A, B, N = 10, 9, 2002

    data = np.random.normal(A, B, N)
    print(compute_statistics(data))
    for m in [10, 20]:
        plot_histogram(data, m)
        plot_polygon(data, m)
        plot_polygon(data, m, cumulative=True)
        plot_polygon(data, m, cumulative=True, normalized=True)

    plot_corridor(data)
    plot_kolmogorov()


if __name__ == "__main__":
    main()
