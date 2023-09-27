# %%
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class NormalDistributionConfidenceInterval:
    def __init__(self, data):
        self.data = data
        self.n = len(data)

        self.mean = np.mean(data)
        self.std_dev = np.std(data, ddof=1)

        # Ступені свободи
        self.df = self.n - 1
        # Стандартна помилка
        self.se = self.std_dev / np.sqrt(self.n)

        self.alpha = 0.05
        self.t = stats.t.ppf(1 - self.alpha, self.df)

        self.ci_mean = [self.mean - self.t * self.se, self.mean + self.t * self.se]
        self.chi2 = [
            stats.chi2.ppf(self.alpha / 2, self.df),
            stats.chi2.ppf(1 - self.alpha / 2, self.df),
        ]
        self.ci_var = [self.df * self.std_dev**2 / chi for chi in self.chi2]

    def __str__(self) -> str:
        return f"""
            границі довірчого інтервалу для мат. сподівання: {self.ci_mean}
            границі довірчого інтервалу для дисперсії: {self.ci_var}
            границі довірчого інтервалу xi-функції: {self.chi2}
        """


class PoissonDistributionConfidenceInterval:
    def __init__(self, data, alpha=0.05):
        self.data = data
        self.alpha = alpha

        self.n = len(data)

        self.max = np.max(data)
        self.min = np.min(data)

        self.range = self.max - self.min

        self.sorted_data = np.sort(data)

    def histogram(self, bins: int = 10):
        delta = self.range / bins
        pk = np.array([self.min + delta / 2 * (2 * j - 1) for j in range(1, bins + 1)])

        fk, _ = np.histogram(self.data, bins=np.linspace(self.min, self.max, bins + 1))
        fk_normalized = fk / self.n

        plt.bar(
            pk,
            fk_normalized,
            width=delta,
            align="center",
            edgecolor="red",
            facecolor="none",
        )
        plt.xlabel("$p_k$")
        plt.ylabel("$\\frac{f_k}{N}$")
        plt.show()

    def lam_est(self, n):
        return 1 / n * np.sum(self.data[:n])

    def lam_left(self, n, xalpha):
        return (math.sqrt(self.lam_est(n)) - xalpha / (2 * math.sqrt(n))) ** 2

    def lam_right(self, n, xalpha):
        return (math.sqrt(self.lam_est(n)) + xalpha / (2 * math.sqrt(n))) ** 2

    def plot_confidence_interval(self):
        xalpha = stats.norm.ppf(1 - self.alpha / 2)
        n = range(50, self.n + 1)

        lam_est = np.array([self.lam_est(n) for n in n])
        lam_left = np.array([self.lam_left(n, xalpha) for n in n])
        lam_right = np.array([self.lam_right(n, xalpha) for n in n])

        plt.plot(n, lam_est, label="$\\lambda est(n)$", color="r", linestyle="--")
        plt.plot(n, lam_left, label="$\\lambda left(n)$", color="b", linestyle="--")
        plt.plot(n, lam_right, label="$\\lambda right(n)$", color="g", linestyle="--")

        plt.legend(loc="upper right")
        plt.xlabel("$n$")
        plt.show()

        lam_diff = lam_right - lam_left
        plt.plot(
            n,
            lam_diff,
            label="$\\lambda right(n) - \\lambda left(n)$",
            color="r",
        )
        plt.legend(loc="upper right")
        plt.show()


def cofidence_interval(xy: list[int, 2], alpha=0.05):
    xalpha = stats.norm.ppf(1 - alpha / 2)

    n = len(xy[0])

    xmean = np.mean(xy[0])
    ymean = np.mean(xy[1])

    m = 1 / n * np.sum((xy[0] - xmean) * (xy[1] - ymean))

    sigma2x = 1 / n * np.sum((xy[0] - xmean) ** 2)
    sigma2y = 1 / n * np.sum((xy[1] - ymean) ** 2)

    k = m / math.sqrt(sigma2x * sigma2y)

    k_formula = (
        np.sum(xy[0] * xy[1]) - 1 / n * (np.sum(xy[0]) * np.sum(xy[1]))
    ) / math.sqrt(
        (np.sum(xy[0] ** 2) - 1 / n * (np.sum(xy[0])) ** 2)
        * (np.sum(xy[1] ** 2) - 1 / n * (np.sum(xy[1])) ** 2)
    )

    assert abs(k-k_formula) < 1e-5

    kleft = math.tanh(math.atanh(k) - xalpha / np.sqrt(n - 3))
    kright = math.tanh(math.atanh(k) + xalpha / np.sqrt(n - 3))

    print(f"Коефіцієнт кореляції: {k}")
    print(f"Границі довірчого інтервалу: [{kleft}, {kright}]")

    plt.plot(xy[0], xy[1], "o", color="red", label="$[(XY^T)^{<2>}]$")
    plt.ylabel("$[(XY^T)^{<2>}]$")
    plt.xlabel("$[(XY^T)^{<1>}]$")
    plt.legend()
    plt.show()


def main():
    np.random.seed(0)

    data = np.random.normal(10, 9, 2002)

    ndci = NormalDistributionConfidenceInterval(data)

    print(ndci)

    data = np.random.poisson(6, 2002)

    pdci = PoissonDistributionConfidenceInterval(data)

    pdci.histogram(bins=10)
    pdci.plot_confidence_interval()

    xy1 = np.array(
        [
            [-2.7, -0.931, -0.257, 1.381, -0.315, -3.05, 0.054, 0.835],
            [-14.902, -18.113, 6.138, 13.813, -0.227, 4.927, 2.576, 1.184],
        ]
    )
    print(xy1)
    cofidence_interval(xy1)

    xy2 = np.array(
        [
            [1.661, 3.333, -1.12, 0.377, -2.28, -5.092, 3.124],
            [-14.433, 1.527, 11.866, 2.121, -6.254, 1.58, 13.972],
        ]
    )
    print(xy2) 
    cofidence_interval(xy2)


if __name__ == "__main__":
    main()

# %%
