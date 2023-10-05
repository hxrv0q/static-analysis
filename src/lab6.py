import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def main():
    data = np.array(
        [
            [5.357, 5.447, 4.929, 5.075, 5.38, 4.305],
            [5.879, 4.683, 6.182, 7.513, 5.183, 6.708],
            [4.206, 3.26, 4.137, 3.806, 4.646, 0.0],
            [4.092, 4.465, 5.852, 5.724, 0.0, 0.0],
        ]
    )

    n = np.count_nonzero(data, axis=1)

    print(f"{n=}")

    m = data.shape[0]
    print(f"{m=}")

    N = np.sum(n)
    print(f"{N=}")

    x = np.sum(data, axis=1) / n
    print(f"{x=}")

    s12 = np.sum((data - x[:, np.newaxis]) ** 2 * (data != 0))
    print(f"s12={s12:.3f}")

    xn = np.mean(data[data != 0])
    print(f"xn={xn:.3f}")

    s22 = np.sum((x - xn) ** 2 * n)
    print(f"s22={s22:.3f}")

    fh = (s22 * (N - m)) / (s12 * (N - 1))
    print(f"fh={fh:.3f}")

    alpha = 0.05
    xalpha = stats.f.ppf(1 - alpha, N - 1, N - m)
    print(f"xalpha={xalpha:.3f}")

    # fmt:off
    if fh > xalpha:
        print("Оскільки, fh > xalpha, то гіпотеза H0 відхиляється. Тобто, тип реклами впливає на обсяг продажу товару")
    else:
        print( "Оскільки, fh < xalpha, то гіпотеза H0 не відхиляється. Тобто, тип реклами не впливає на обсяг продажу товару")
    # fmt:on

    s2 = np.sum((data - xn) ** 2 * (data != 0))
    r2 = s22 / s2

    print(f"r2 = {r2:.3f}")

    a = x
    print(f"{a=}")

    sigma2 = s12 / (N - m)
    sigma = np.sqrt(sigma2)
    print(f"sigma={sigma:.3f}")

    labels = ["A", "B", "C", "D"]

    plt.figure(figsize=(10, 6))

    t = np.linspace(np.min(data[data > 0]) - 1, np.max(data) + 1, 1000)

    # fmt:off
    for i, lable in enumerate(labels):
        linestyle = "--" if i % 2 == 1 else "-"
        plt.plot(t, stats.norm.pdf(t, a[i], sigma), label=f"Тип реклами {lable}", linestyle=linestyle)
    # fmt:on

    plt.legend()
    plt.show()

    s = np.sum((data - xn) ** 2 * (data != 0)) / (n - 1)
    print(f"{s=}")

    S = np.sum(s * (n - 1)) / np.sum(n - 1)
    print(f"S={S:.3f}")

    qprod = np.prod(n - 1)
    qsum = np.sum(1 / (n - 1))
    q = 1 / (1 - (1 / (3 * (m - 1)) * (qsum + 1 / qprod)))
    print(f"q={q:.3f}")

    b = q * np.sum((n - 1) * np.log(S / s))
    print(f"b={b:.3f}")

    xalpha = stats.chi2.ppf(1 - alpha, m - 1)
    print(f"xalpha={xalpha:.3f}")

    # fmt:off
    if b < xalpha:
        print("Оскільки (за критерієм Бартлета) b < xalpha, то гіпотеза про рівність дисперсій приймається")
    else:
        print("Оскільки (за критерієм Бартлета) b > xalpha, то гіпотеза про рівність дисперсій відхиляється")
    # fmt:on


if __name__ == "__main__":
    main()
