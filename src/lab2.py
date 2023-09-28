import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def generate_data(dist_type: str, N: int, **kwargs) -> np.ndarray:
    match dist_type:
        case "normal":
            return np.random.normal(kwargs["A"], kwargs["B"], N)
        case "poisson":
            return np.random.poisson(kwargs["lam"], N)
        case "exponential":
            return np.random.exponential(1 / kwargs["lam"], N)
        case _:
            raise ValueError(f"Unknown distribution type: {dist_type}")


def find_unbiased_estimates(data: np.ndarray) -> tuple[float, float]:
    n = len(data)
    expectation = np.sum(data) / n
    variance = np.sum((data - expectation) ** 2) / n
    corrected_variance = n / (n - 1) * variance
    return expectation, corrected_variance


def plot_graph(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    ax: None | float = None,
) -> None:
    if ax is None:
        plt.plot(x, y, color="r", label=title)
    else:
        plt.scatter(x, y, label=title, marker="o", s=2)
        plt.axhline(y=ax, color="r", linestyle="-", label=f"{ax}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def compute_log_likehood(
    lams: np.ndarray, data: np.ndarray, log_likehood_func: Callable
) -> np.ndarray:
    return np.array([log_likehood_func(lam, data) for lam in lams])


def compute_lambda(n: int, data: np.ndarray, lambda_func: Callable) -> np.ndarray:
    return np.array([lambda_func(i, data) for i in range(1, n + 1)])


def print_lambda_values(lambda_values: np.ndarray, intervals: list[int]) -> None:
    for i in intervals:
        print(f"λ({i}) = {lambda_values[i - 1]}")


def plot_and_print(
    data: np.ndarray,
    lam: float,
    lams: np.ndarray,
    log_likehood_func: Callable,
    lambda_func: Callable,
    intervals: list[int],
) -> None:
    n = len(data)
    lambda_values = compute_lambda(n, data, lambda_func)
    log_likehood_values = compute_log_likehood(lams, data, log_likehood_func)

    plot_graph(
        lams, log_likehood_values, "$\\lambda$", "$LnL(\\lambda)$", "$LnL(\\lambda)$"
    )
    plot_graph(
        np.arange(1, n + 1),
        lambda_values,
        "n",
        "$\\lambda(n)$",
        "$\\lambda(n)$",
        ax=lam,
    )
    print_lambda_values(lambda_values, intervals)


def main():
    np.random.seed(0)

    N = 2002
    lam = 0.6
    lams = np.arange(0.1, 1, 0.01)
    intevals = [i * 6 for i in range(10, 6 * 10, 10)]

    data_normal = generate_data("normal", N, A=10, B=9)
    expectation, corrected_variance = find_unbiased_estimates(data_normal)
    print(f"Математичне сподівання: {expectation}")
    print(f"Виправлена дисперсія: {corrected_variance}")

    data_poisson = generate_data("poisson", N, lam=lam)

    def poisson_log_likehood_func(lam, data):
        return -len(data) * lam + np.sum(data) * np.log(lam) - np.sum(data)

    def poisson_lambda_func(n, data):
        return np.sum(data[:n]) / n

    plot_and_print(
        data_poisson,
        lam,
        lams,
        poisson_log_likehood_func,
        poisson_lambda_func,
        intevals,
    )

    data_expotential = generate_data("exponential", N, lam=lam)

    def exponential_log_likehood_func(lam, data):
        return len(data) * math.log(lam) - lam * np.sum(data)

    def exponential_lambda_func(n, data):
        return n / np.sum(data[:n])

    plot_and_print(
        data_expotential,
        lam,
        lams,
        exponential_log_likehood_func,
        exponential_lambda_func,
        intevals,
    )


if __name__ == "__main__":
    main()
