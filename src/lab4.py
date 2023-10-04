import numpy as np
from scipy.stats import norm, t, chi2


def test_mean_know_variance(data, expectation, variance, alphas):
    mean = np.mean(data)
    phi = (mean - expectation) / np.sqrt(variance / len(data))

    for alpha in alphas:
        z_alpha = norm.ppf(1 - alpha / 2)

        if phi > z_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H1 (μ > a): відхиляємо нульову гіпотезу, оскільки φ={phi} > z_α={z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H1 (μ > a): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < z_α={z_alpha}"
            )
        if phi < -z_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H2 (μ < a): відхиляємо нульову гіпотезу, оскільки φ={phi} < -z_α={-z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H2 (μ < a): не можемо відхилити нульову гіпотезу, оскільки φ={phi} > -z_α={-z_alpha}"
            )

        if abs(phi) > z_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H3 (μ ≠ a): відхиляємо нульову гіпотезу, оскільки |φ|={abs(phi)} > z_α={z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H3 (μ ≠ a): не можемо відхилити нульову гіпотезу, оскільки |φ|={abs(phi)} < z_α={z_alpha}"
            )


def test_mean_unknown_variance(data, expectation, alphas):
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    phi = (mean - expectation) / (std / np.sqrt(len(data)))
    df = len(data) - 1

    for alpha in alphas:
        t_alpha = t.ppf(1 - alpha / 2, df)

        if phi > t_alpha:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H1 (μ > a): відхиляємо нульову гіпотезу, оскільки φ={phi} > t_α={t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H1  (μ > a): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < t_α={t_alpha}"
            )

        if phi < -t_alpha:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H2 (μ < a): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < -t_α={-t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H2 (μ < a): відхиляємо нульову гіпотезу, оскільки φ={phi} > -t_α={-t_alpha}"
            )

        if abs(phi) > t_alpha:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H3 (μ ≠ a): відхиляємо нульову гіпотезу, оскільки |φ|={abs(phi)} > t_α={t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H3 (μ ≠ a): не можемо відхилити нульову гіпотезу, оскільки |φ|={abs(phi)} < t_α={t_alpha}"
            )


def test_variance(data, variance, alphas):
    var = np.var(data, ddof=1)
    phi = (len(data) - 1) * var / variance
    df = len(data) - 1

    for alpha in alphas:
        chi2_alpha_left = chi2.ppf(alpha / 2, df)
        chi2_alpha_right = chi2.ppf(1 - alpha / 2, df)

        if phi > chi2_alpha_right:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H1 (σ^2 > var): відхиляємо нульову гіпотезу, оскільки φ={phi} > χ^2_α/2={chi2_alpha_right}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H1 (σ^2 > var): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < χ^2_α/2={chi2_alpha_right}"
            )

        if phi < chi2_alpha_left:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H2 (σ^2 < var): відхиляємо нульову гіпотезу, оскільки φ={phi} < χ^2_1-α/2={chi2_alpha_left}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H2 (σ^2 < var): не можемо відхилити нульову гіпотезу, оскільки φ={phi} > χ^2_1-α/2={chi2_alpha_left}"
            )

        if phi < chi2_alpha_left or phi > chi2_alpha_right:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H3 (σ^2 ≠ var): відхиляємо нульову гіпотезу, оскільки φ={phi} не належить [{chi2_alpha_left}, {chi2_alpha_right}]"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивна гіпотеза H3 (σ^2 ≠ var): не можемо відхилити нульову гіпотезу, оскільки φ={phi} належить [{chi2_alpha_left}, {chi2_alpha_right}]"
            )


def test_two_means_known_variance(data1, data2, var1, var2, alphas):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    M, N = len(data1), len(data2)
    phi = (mean1 - mean2) / np.sqrt(var1 / M + var2 / N)

    for alpha in alphas:
        z_alpha = norm.ppf(1 - alpha / 2)

        if phi > z_alpha:
            print(
                f"Для α={alpha} та альтернативної гіпотези H1 (μ1 > μ2): відхиляємо нульову гіпотезу, оскільки φ={phi} > z_α={z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернативної гіпотези H1 (μ1 > μ2): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < z_α={z_alpha}"
            )

        if phi < -z_alpha:
            print(
                f"Для α={alpha} та альтернативної гіпотези H2 (μ1 < μ2): відхиляємо нульову гіпотезу, оскільки φ={phi} < -z_α={-z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернативної гіпотези H2 (μ1 < μ2): не можемо відхилити нульову гіпотезу, оскільки φ={phi} > -z_α={-z_alpha}"
            )

        if abs(phi) > z_alpha:
            print(
                f"Для α={alpha} та альтернативної гіпотези H3 (μ1 ≠ μ2): відхиляємо нульову гіпотезу, оскільки |φ|={abs(phi)} > z_α={z_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернативної гіпотези H3 (μ1 ≠ μ2): не можемо відхилити нульову гіпотезу, оскільки |φ|={abs(phi)} < z_α={z_alpha}"
            )


def test_two_means_unknown_variance(data1, data2, alphas):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    M, N = len(data1), len(data2)

    df = N + M - 2
    phi = (mean1 - mean2) / np.sqrt(
        (1 / N + 1 / M) * ((M - 1) * var1 + (N - 1) * var2) / df
    )
    for alpha in alphas:
        t_alpha = t.ppf(1 - alpha / 2, df)

        if phi > t_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H1 (μ1 > μ2): відхиляємо нульову гіпотезу, оскільки φ={phi} > t_α={t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H1 (μ1 > μ2): не можемо відхилити нульову гіпотезу, оскільки φ={phi} < t_α={t_alpha}"
            )

        if phi < -t_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H2 (μ1 < μ2): відхиляємо нульову гіпотезу, оскільки φ={phi} < -t_α={-t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H2 (μ1 < μ2): не можемо відхилити нульову гіпотезу, оскільки φ={phi} > -t_α={-t_alpha}"
            )

        if abs(phi) > t_alpha:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H3 (μ1 ≠ μ2): відхиляємо нульову гіпотезу, оскільки |φ|={abs(phi)} > t_α={t_alpha}"
            )
        else:
            print(
                f"Для α={alpha} та альтернетивної гіпотези H3 (μ1 ≠ μ2): не можемо відхилити нульову гіпотезу, оскільки |φ|={abs(phi)} < t_α={t_alpha}"
            )


def main():
    np.random.seed(0)

    a = 9
    sigma2 = 10
    M = 2002

    data = np.random.normal(a, np.sqrt(sigma2), M)

    alphas = [0.05, 0.1]

    print(
        "Перевірка альтернативних гіпотез про математичне сподівання (відома дисперсія)"
    )
    test_mean_know_variance(data, a, sigma2, alphas)

    print(
        "Перевірка альтернативних гіпотез про математичне сподівання (невідома дисперсія)"
    )
    test_mean_unknown_variance(data, a, alphas)

    print("Перевірка альтернативних гіпотез про дисперсію")
    test_variance(data, sigma2, alphas)

    sigma2y = sigma2 + 2
    N = M - 100
    data2 = np.random.normal(a, np.sqrt(sigma2y), N)

    print(
        "Перевірка альтернативних гіпотез про рівність математичних сподівань (відома дисперсія)"
    )
    test_two_means_known_variance(data, data2, sigma2, sigma2y, alphas)

    print(
        "Перевірка альтернативних гіпотез про рівність математичних сподівань (невідома дисперсія)"
    )
    test_two_means_unknown_variance(data, data2, alphas)


if __name__ == "__main__":
    main()
