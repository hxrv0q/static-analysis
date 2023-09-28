# %%
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class ConfidenceIntervalAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.xalpha = stats.norm.ppf(1 - alpha / 2)

    def compute_cofidence_interval_normal(self, data):
        N = len(data)
        t = stats.t.ppf(1 - self.alpha, N)
        mx = np.mean(data)
        Dx = np.var(data)

        ci_mx = (mx - t * Dx / np.sqrt(N), mx + t * Dx / np.sqrt(N))
        ci_Dx = (
            (N - 1) * Dx / stats.chi2.ppf(1 - self.alpha / 2, N - 1),
            (N - 1) * Dx / stats.chi2.ppf(self.alpha / 2, N - 1),
        )

        return ci_mx, ci_Dx

    def plot_histogram(self, data, bins, normalized=True):
        pmin, pmax = np.min(data), np.max(data)
        R = pmax - pmin
        delta = R / bins

        pk = np.array([pmin + delta / 2 * (2 * j - 1) for j in range(1, bins)])

        fk, _ = np.histogram(data, bins=np.linspace(pmin, pmax, bins))
        if normalized:
            fk = fk / len(data)

        plt.bar(pk, fk, width=delta, align="center", edgecolor="r", facecolor="none")
        plt.show()

    def estimate_lambda(self, data, n):
        return 1 / n * np.sum(data[:n])

    def compute_lambda_intervals(self, data, n_range):
        lam_est = np.array([self.estimate_lambda(data, n) for n in n_range])
        lam_left = np.array(
            [
                (math.sqrt(lam) - self.xalpha / (2 * math.sqrt(i))) ** 2
                for lam, i in zip(lam_est, n_range)
            ]
        )
        lam_right = np.array(
            [
                (math.sqrt(lam) + self.xalpha / (2 * math.sqrt(i))) ** 2
                for lam, i in zip(lam_est, n_range)
            ]
        )
        return lam_est, lam_left, lam_right

    def plot_lambda_intervals(self, n_range, lam_est, lam_left, lam_right):
        plt.plot(n_range, lam_est, label="$\\lambda est(n)$", color="r", linestyle="--")
        plt.plot(
            n_range, lam_left, label="$\\lambda left(n)$", color="g", linestyle="--"
        )
        plt.plot(
            n_range, lam_right, label="$\\lambda right(n)$", color="b", linestyle="--"
        )
        plt.legend()
        plt.xlabel("$n$")
        plt.show()

        lam = lam_right - lam_left
        plt.plot(
            n_range, lam, label="$\\lambda right(n) - \\lambda left(n)$", color="r"
        )
        plt.legend()
        plt.show()

    def compute_and_plot_correlation_interval(self, xy):
        n = len(xy[0])
        xmean, ymean = np.mean(xy, axis=1)
        m = 1 / n * np.sum((xy[0] - xmean) * (xy[1] - ymean))

        sigma2x = 1 / n * np.sum((xy[0] - xmean) ** 2)
        sigma2y = 1 / n * np.sum((xy[1] - ymean) ** 2)
        k = m / math.sqrt(sigma2x * sigma2y)

        k_formula = (
            np.sum(xy[0] * xy[1]) - 1 / n * (np.sum(xy[0]) * np.sum(xy[1]))
        ) / math.sqrt(
            (np.sum(xy[0] ** 2) - 1 / n * (np.sum(xy[0])) ** 2)
            * (np.sum(xy[1] ** 2) - 1 / n * np.sum(xy[1]) ** 2)
        )

        assert abs(k - k_formula) < 1e-5

        kleft = math.tanh(math.atanh(k) - self.xalpha / math.sqrt(n - 3))
        kright = math.tanh(math.atanh(k) + self.xalpha / math.sqrt(n - 3))

        plt.scatter(xy[0], xy[1], color="r")
        plt.show()

        return kleft, k, kright


def main():
    analyzer = ConfidenceIntervalAnalyzer()

    A, B, N = 10, 9, 2002

    np.random.seed(0)

    normal_data = np.random.normal(A, B, N)
    ci_mx, ci_var = analyzer.compute_cofidence_interval_normal(normal_data)
    print(f"Довірчий інтервал для мат. сподівання: {ci_mx}")
    print(f"Довірчий інтервал для дисперсії: {ci_var}")

    # Розподіл Пуассона
    lam = 6
    poisson_data = np.random.poisson(lam, N)
    analyzer.plot_histogram(poisson_data, 10)
    n_range = range(50, N)
    l_est, l_left, l_right = analyzer.compute_lambda_intervals(poisson_data, n_range)
    analyzer.plot_lambda_intervals(n_range, l_est, l_left, l_right)

    # Коефіцієнт кореляції
    xy = np.array(
        [
            [-2.7, -0.931, -0.257, 1.381, -0.315, -3.05, 0.054, 0.835],
            [-14.902, -18.113, 6.138, 13.813, -0.227, 4.927, 2.576, 1.184],
        ]
    )
    kleft, k, kright = analyzer.compute_and_plot_correlation_interval(xy)
    print(f"Коефіцієнт кореляції: {k}")
    print(f"Границі довірчого інтервалу: [{kleft}, {kright}]")

    xy = np.array(
        [
            [1.661, 3.333, -1.12, 0.377, -2.28, -5.092, 3.124],
            [-14.433, 1.527, 11.866, 2.121, -6.254, 1.58, 13.972],
        ]
    )
    kleft, k, kright = analyzer.compute_and_plot_correlation_interval(xy)
    print(f"Коефіцієнт кореляції: {k}")
    print(f"Границі довірчого інтервалу: [{kleft}, {kright}]")


if __name__ == "__main__":
    main()

# %%
