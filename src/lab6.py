# %%
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    np.random.seed(0)

    data = {
        "A": [5.357, 5.447, 4.929, 5.075, 5.38, 4.305],
        "B": [5.879, 4.683, 6.182, 7.513, 5.183, 6.708],
        "C": [4.206, 3.26, 4.137, 3.806, 4.646],
        "D": [4.092, 4.465, 5.852, 5.724],
    }

    n = sum(len(data[key]) for key in data)
    k = len(data)

    alpha = 0.05
    x_alpha = stats.f.ppf(1 - alpha, k - 1, n - k)

    f_stat, p_value = stats.f_oneway(data["A"], data["B"], data["C"], data["D"])

    print(f"{f_stat=} > {x_alpha=}: {f_stat > x_alpha}")

    means = {key: np.mean(data[key]) for key in data}
    std_devs = {key: np.std(data[key], ddof=1) for key in data}

    # Plotting
    x = np.linspace(
        min([min(values) for values in data.values()]) - 1,
        max([max(values) for values in data.values()]) + 1,
        1000,
    )

    plt.figure(figsize=(10, 6))
    for key in data:
        plt.plot(
            x,
            stats.norm.pdf(x, means[key], std_devs[key]),
            label=f"Type {key}",
            linestyle="-" if key in ["A", "C"] else "--",
            linewidth=2,
        )

    plt.xlabel("t")
    plt.legend()
    plt.grid(True)
    plt.show()

    result = stats.bartlett(data["A"], data["B"], data["C"], data["D"])

    print(f"{result.statistic=} < {x_alpha=}: {result.statistic < x_alpha}")


if __name__ == "__main__":
    main()

# %%
