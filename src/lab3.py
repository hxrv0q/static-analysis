import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def compute_confidence_interval_variance_and_expectation(
    data: np.ndarray, alpha: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    n = len(data)
    t = stats.t.ppf(1 - alpha, n)
    expectation, variance = np.mean(data), np.var(data)

    ci_expectation = (
        expectation - t * variance / np.sqrt(n),
        expectation + t * variance / np.sqrt(n),
    )
    ci_variance = (
        float((n - 1) * expectation / stats.chi2.ppf(1 - alpha / 2, n - 1)),
        float((n - 1) * expectation / stats.chi2.ppf(alpha / 2, n - 1)),
    )

    return ci_expectation, ci_variance


def plot_histogram(data: np.ndarray, bins: int, normalized: bool = True) -> None:
    pmin, pmax, delta = np.min(data), np.max(data), np.ptp(data) / bins
    # pj = pmin + delta / 2 * (2 * j - 1)
    # pj = pmin + delta * (j - 0.5), j = 1, ..., m + 1
    pk = pmin + delta * (np.arange(bins) - 0.5)

    fk, _ = np.histogram(data, bins=np.linspace(pmin, pmax, bins + 1))
    fk = fk / len(data) if normalized else fk
    plt.bar(pk, fk, width=delta, align="center", edgecolor="r", facecolor="none")
    plt.show()


def estimate_lambda(data: np.ndarray, n: int) -> float:
    return np.mean(data[:n])


def compute_lambda_intervals(
    data: np.ndarray, n_range: range, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xalpha = stats.norm.ppf(1 - alpha / 2)
    lam_est = np.array([estimate_lambda(data, n) for n in n_range])
    sqrt_lam_est = np.sqrt(lam_est)
    sqrt_n_range = np.sqrt(n_range)
    lam_left = (sqrt_lam_est - xalpha / (2 * sqrt_n_range)) ** 2
    lam_right = (sqrt_lam_est + xalpha / (2 * sqrt_n_range)) ** 2
    return lam_est, lam_left, lam_right


def plot_lambda_intervals(
    n_range: range, lam_est: np.ndarray, lam_left: np.ndarray, lam_right: np.ndarray
) -> None:
    plt.plot(n_range, lam_est, label="$\\lambda est(n)$", color="r", linestyle="--")
    plt.plot(n_range, lam_left, label="$\\lambda left(n)$", color="g", linestyle="--")
    plt.plot(n_range, lam_right, label="$\\lambda right(n)$", color="b", linestyle="--")
    plt.legend()
    plt.xlabel("$n$")
    plt.show()
    plt.plot(
        n_range,
        lam_right - lam_left,
        label="$\\lambda right(n) - \\lambda left(n)$",
        color="r",
    )
    plt.legend()
    plt.show()


def compute_and_plot_correlation_interval(
    xy: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    n, (xmean, ymean) = len(xy[0]), np.mean(xy, axis=1)
    m = np.mean((xy[0] - xmean) * (xy[1] - ymean))
    sigma2x, sigma2y = np.var(xy, axis=1)
    k = m / math.sqrt(sigma2x * sigma2y)

    kleft = math.tanh(math.atanh(k) - stats.norm.ppf(1 - alpha / 2) / math.sqrt(n - 3))
    kright = math.tanh(math.atanh(k) + stats.norm.ppf(1 - alpha / 2) / math.sqrt(n - 3))

    plt.scatter(xy[0], xy[1], color="r")
    plt.show()

    return kleft, k, kright


def main():
    np.random.seed(0)
    A, B, N, alpha, lam, bins = 10, 9, 2002, 0.05, 6, 10

    normal_data = np.random.normal(A, B, N)
    ci_expectation, ci_variance = compute_confidence_interval_variance_and_expectation(
        normal_data, alpha
    )
    print(f"Довірчий інтервал для мат. сподівання: {ci_expectation}")
    print(f"Довірчий інтервал для дисперсії: {ci_variance}")

    # Розподіл Пуассона
    poisson_data = np.random.poisson(lam, N)
    plot_histogram(poisson_data, bins)
    lam_est, lam_left, lam_right = compute_lambda_intervals(
        poisson_data, range(50, N), alpha
    )
    plot_lambda_intervals(range(50, N), lam_est, lam_left, lam_right)

    for xy in [
        [
            [-2.7, -0.931, -0.257, 1.381, -0.315, -3.05, 0.054, 0.835],
            [-14.902, -18.113, 6.138, 13.813, -0.227, 4.927, 2.576, 1.184],
        ],
        [
            [1.661, 3.333, -1.12, 0.377, -2.28, -5.092, 3.124],
            [-14.433, 1.527, 11.866, 2.121, -6.254, 1.58, 13.972],
        ],
    ]:
        k_left, k, k_right = compute_and_plot_correlation_interval(np.array(xy))
        print(f"Коефіцієнт кореляції: {k}")
        print(f"Границі довірчого інтервалу: [{k_left}, {k_right}]")


if __name__ == "__main__":
    main()
