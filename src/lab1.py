# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class SampleDistribution:
    def __init__(self, data):
        self.data = data

        self.min = np.min(data)
        self.max = np.max(data)
        self.range = self.max - self.min
        self.n = len(data)

        self.mean = np.mean(data)
        self.var = np.var(data)
        self.std = np.std(data)
        self.skew = stats.skew(data)
        self.kurtosis = stats.kurtosis(data)

    def plot(self, bins: int):
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        self.histogram(bins)

        plt.subplot(1, 4, 2)
        self.polygon(bins, f"Полігон m={bins}")

        plt.subplot(1, 4, 3)
        self.polygon(
            bins, cumulative=True, title=f"Полігон накопичених частот m={bins}"
        )

        plt.subplot(1, 4, 4)
        self.polygon(
            bins,
            cumulative=True,
            normalized=True,
            title=f"Нормований полігон накопичених частот m={bins}",
        )

        plt.show()

    def histogram(self, bins: int):
        plt.hist(
            self.data,
            bins=bins,
            label="$f_k$",
            range=(self.min, self.max),
            alpha=0.7,
            edgecolor="black",
        )
        plt.title(f"Гістограма m={bins}")
        plt.legend()

    def polygon(
        self, bins: int, title: str, cumulative: bool = False, normalized: bool = False
    ):
        frequency, bin_edges = np.histogram(self.data, bins=bins)
        if normalized:
            frequency = frequency / np.sum(frequency)
        if cumulative:
            frequency = np.cumsum(frequency)

        bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(bin_mids, frequency, marker="o", linestyle="-")
        plt.title(title)
        plt.xlabel("x")

    def corridor(self, alpha=0.05):
        sorted_data = np.sort(self.data)
        n = len(sorted_data)

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        ci = z_alpha / np.sqrt(n)

        f = np.arange(1, n + 1) / n
        lower_bound = f - ci
        upper_bound = f + ci

        plt.plot(
            sorted_data,
            lower_bound,
            linestyle="--",
            color="red",
            label="$F_k - \\frac{z_{\\alpha}}{\\sqrt{n}}$",
        )
        plt.plot(
            sorted_data,
            upper_bound,
            linestyle="--",
            color="green",
            label="$F_k + \\frac{z_{\\alpha}}{\\sqrt{n}}$",
        )
        plt.xlabel("x")
        plt.legend()

        plt.show()

    def __str__(self):
        return f"""
            Максимальне значення: {self.max}\n
            Мінімальне значення: {self.min}\n
            Розмах вибірки: {self.range}\n
            Кількість елементів: {self.n}\n
            Середнє значення: {self.mean}\n
            Дисперсія: {self.var}\n
            Середнє квадратичне відхилення: {self.std}\n
            Коефіцієнт асиметрії: {self.skew}\n
            Коефіцієнт ексцесу: {self.kurtosis}\n
        """


def N(z, eps=1e-3):
    return math.floor((1.0 / z) * np.sqrt(0.5 * np.log(1.0 / eps))) + 1


def K(z, eps=1e-3):
    if z <= 0:
        return 0

    n = N(z, eps)

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


if __name__ == "__main__":
    np.random.seed(0)

    data = np.random.normal(10, 9, 2002)
    dist = SampleDistribution(data)

    print(dist)

    m = [10, 20]
    for m1 in m:
        dist.plot(m1)

    dist.corridor()

    z = np.linspace(0.1, 2, 500)

    eps = 1e-3

    k = [K(z1, eps) for z1 in z]

    alpha = 0.05
    plot(z, k, "z", "$K(z)$", "Критерій Колмогорова", axhline=1 - alpha)

    n = [N(z1, eps) for z1 in z]

    plot(z, n, "z", "$N(z)$", f"$N(z)$ при $\\varepsilon=${eps}")

# %%
